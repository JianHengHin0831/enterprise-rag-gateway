"""M2 pipeline features on the real retrieval path (in-memory Qdrant + FakeEmbedder):
L1/L2 cache, budget reserve/settle/downgrade, query rewrite, and /v1/chat SSE."""

import asyncio
import json

import pytest
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient

from app.cache.keys import index_version_key
from app.cache.layered import LayeredCache
from app.config import get_settings
from app.cost.budget import BudgetExceededError, BudgetGuard, budget_key
from app.cost.tokens import estimate_messages_tokens
from app.gateway.base import (
    ChatMessage,
    LLMResponse,
    NonRetryableLLMError,
    RetryableLLMError,
    StreamChunk,
    Usage,
)
from app.gateway.providers.fake import FakeLLM
from app.gateway.providers.template import TemplateProvider
from app.gateway.router import ModelRouter
from app.main import app, get_pipeline
from app.observability.trace import start_trace
from app.rag.embeddings import FakeEmbedder
from app.rag.pipeline import RagPipeline
from app.rag.prompts import load_prompt
from app.rag.retriever import HybridRetriever
from app.rag.rewrite import QueryRewriter
from app.schemas import ChatRequest, ChatTurn, Identity, Role
from tests.conftest import TEST_COLLECTION, direct_vendor_models_config
from tests.test_streaming import parse_sse

QUESTION = "What is the P6 salary band at Acme?"


class CountingLLM(FakeLLM):
    """Fixed usage per call (so budget arithmetic is exact) and recorded prompts."""

    def __init__(self, name: str = "dashscope", *, usage: int = 100, **kw) -> None:
        super().__init__(name, **kw)
        self.usage = usage
        self.messages: list[list[ChatMessage]] = []

    async def chat(self, messages, *, model, **kw) -> LLMResponse:
        self.messages.append(messages)
        r = await super().chat(messages, model=model, **kw)
        return r.model_copy(update={"usage": Usage(prompt_tokens=self.usage, completion_tokens=0)})

    async def stream(self, messages, *, model, **kw):
        self.messages.append(messages)
        async for c in super().stream(messages, model=model, **kw):
            if c.usage is not None:
                c = c.model_copy(update={"usage": Usage(prompt_tokens=self.usage)})
            yield c


def _config(settings, **budget) -> dict:
    cfg = direct_vendor_models_config(settings.models_config_path)
    cfg["retry"] = {**cfg["retry"], "base_delay_s": 0, "max_delay_s": 0}
    if budget:
        cfg["budget"] = {**cfg["budget"], **budget}
    return cfg


@pytest.fixture
def redis():
    return FakeAsyncRedis()


@pytest.fixture
def build(settings, indexed_qdrant, redis):
    """build(llm, cache=True, budget=None, rewriter_llm=None, **budget_cfg) -> RagPipeline.
    `dashscope` serves both qwen-plus (answer) and qwen-turbo (rewrite)."""

    def _build(llm, *, cache=True, budget_limits=None, on_exceeded="reject", rewrite=False):
        cfg = _config(settings)
        if budget_limits is not None:
            cfg["budget"] = {"daily_tokens_per_tenant": budget_limits, "on_exceeded": on_exceeded}
        router = ModelRouter(config=cfg, providers={"dashscope": llm, "local": TemplateProvider()})
        guard = BudgetGuard.from_config(redis, cfg) if budget_limits is not None else None
        rewriter = (
            QueryRewriter(
                router=router,
                prompt=load_prompt(settings.prompts_dir, "rewrite", 1),
                timeout_s=1.0,
                min_words=4,
                budget=guard,
            )
            if rewrite
            else None
        )
        return RagPipeline(
            retriever=HybridRetriever(indexed_qdrant, FakeEmbedder(), TEST_COLLECTION),
            router=router,
            prompt=load_prompt(settings.prompts_dir, "rag_qa", 1),
            retrieve_top_k=20,
            context_top_n=5,
            budget=guard,
            cache=(LayeredCache(redis, l1_size=500, l1_ttl_s=60, l2_ttl_s=3600) if cache else None),
            rewriter=rewriter,
        )

    return _build


# ---------- cache ----------


async def test_cache_hit_on_second_identical_question(build, hr):
    llm = CountingLLM(default="P6 is 8,300 to 12,600 USD [1].")
    pipeline = build(llm)
    start_trace()
    first = await pipeline.answer(ChatRequest(query=QUESTION), hr)
    tid = start_trace()
    second = await pipeline.answer(ChatRequest(query="  what is the p6 salary band at acme  "), hr)
    assert not first.cache_hit and second.cache_hit
    assert second.cost_usd == 0 and second.trace_id == tid
    assert second.answer == first.answer and second.citations == first.citations
    assert len(llm.messages) == 1


async def test_l2_hit_backfills_a_fresh_process(build, hr, redis):
    llm = CountingLLM(default="Answer [1].")
    await build(llm).answer(ChatRequest(query=QUESTION), hr)
    other_process = build(llm)  # new L1, same Redis
    resp = await other_process.answer(ChatRequest(query=QUESTION), hr)
    assert resp.cache_hit and len(llm.messages) == 1


async def test_cache_is_isolated_per_role_and_tenant(build, hr, intern):
    llm = CountingLLM(default="Answer [1].")
    pipeline = build(llm)
    await pipeline.answer(ChatRequest(query=QUESTION), hr)
    other_role = await pipeline.answer(ChatRequest(query=QUESTION), intern)
    beta = Identity(tenant_id="beta_inc", role=Role.HR, user_id="u")
    other_tenant = await pipeline.answer(ChatRequest(query=QUESTION), beta)
    assert not other_role.cache_hit and not other_tenant.cache_hit
    assert len(llm.messages) == 3


async def test_degraded_answer_is_not_cached(build, hr):
    llm = CountingLLM(script=[RetryableLLMError("503", status_code=503)] * 3)
    pipeline = build(llm)
    first = await pipeline.answer(ChatRequest(query=QUESTION), hr)
    assert first.degraded and first.model_used == "fallback-template"
    second = await pipeline.answer(ChatRequest(query=QUESTION), hr)
    assert not second.cache_hit and len(llm.messages) == 4


async def test_index_version_bump_invalidates_after_l1_ttl(build, hr, redis):
    llm = CountingLLM(default="Answer [1].")
    pipeline = build(llm)
    await pipeline.answer(ChatRequest(query=QUESTION), hr)
    await redis.set(index_version_key("acme_corp"), 99)  # a re-ingest
    pipeline._cache._versions.clear()  # stands in for the 5 s version TTL elapsing
    resp = await pipeline.answer(ChatRequest(query=QUESTION), hr)
    assert not resp.cache_hit and len(llm.messages) == 2


async def test_multi_turn_without_rewrite_is_not_cached(build, hr):
    llm = CountingLLM(default="Answer [1].")
    pipeline = build(llm)  # no rewriter
    req = ChatRequest(
        query=QUESTION,
        history=[ChatTurn(role="user", content="hi"), ChatTurn(role="assistant", content="hello")],
    )
    await pipeline.answer(req, hr)
    resp = await pipeline.answer(req, hr)
    assert not resp.cache_hit and len(llm.messages) == 2


async def test_cache_redis_failure_is_a_miss_not_an_error(hr):
    class Broken(FakeAsyncRedis):
        async def get(self, *a, **kw):
            from redis.exceptions import ConnectionError

            raise ConnectionError("down")

    cache = LayeredCache(Broken(), l1_size=10, l1_ttl_s=60, l2_ttl_s=60)
    assert await cache.get("k") is None
    assert await cache.index_version("acme_corp") is None


# ---------- budget ----------


async def test_reserve_rejects_and_rolls_back_over_limit(redis):
    guard = BudgetGuard(redis, {"acme_corp": 1000})
    ok = await guard.reserve("acme_corp", 800)
    over = await guard.reserve("acme_corp", 300)
    assert ok.accepted and not over.accepted
    assert await guard.spent("acme_corp") == 800  # the refused reservation was undone
    await guard.settle(ok, 500)
    assert await guard.spent("acme_corp") == 500
    assert await redis.ttl(budget_key("acme_corp")) > 0


async def test_default_limit_and_unlimited_tenants(redis):
    guard = BudgetGuard(redis, {"default": 10})
    assert not (await guard.reserve("new_tenant", 11)).accepted
    unlimited = BudgetGuard(redis, {})
    assert (await unlimited.reserve("t", 10**9)).accepted


async def test_settle_zero_refunds(redis):
    guard = BudgetGuard(redis, {"acme_corp": 1000})
    r = await guard.reserve("acme_corp", 600)
    await guard.settle(r, 0)
    assert await guard.spent("acme_corp") == 0


async def test_concurrent_reservations_never_exceed_budget_plus_one_request(redis):
    limit, estimate = 1_000, 130
    guard = BudgetGuard(redis, {"acme_corp": limit})

    async def request(actual: int) -> bool:
        r = await guard.reserve("acme_corp", estimate)
        if not r.accepted:
            return False
        await asyncio.sleep(0)  # interleave with the other requests
        await guard.settle(r, actual)
        return True

    results = await asyncio.gather(*(request(estimate - i) for i in range(20)))
    assert 0 < sum(results) < 20
    assert await guard.spent("acme_corp") <= limit + estimate


async def test_pipeline_20_concurrent_requests_at_the_limit(build, hr):
    """Acceptance: actual spend never exceeds the daily budget + one request's maximum."""
    llm = CountingLLM(default="Answer [1].", usage=100)
    pipeline = build(llm, cache=False, budget_limits={"acme_corp": 5_000})
    results = await asyncio.gather(
        *(pipeline.answer(ChatRequest(query=f"{QUESTION} #{i}"), hr) for i in range(20)),
        return_exceptions=True,
    )
    served = [r for r in results if not isinstance(r, BaseException)]
    rejected = [r for r in results if isinstance(r, BudgetExceededError)]
    assert len(served) + len(rejected) == 20 and served and rejected
    spent = int(await pipeline._budget._redis.get(budget_key("acme_corp")))
    assert spent == 100 * len(served)  # settled to actual usage
    assert spent <= 5_000 + 100  # budget + one request's maximum


async def test_budget_downgrades_to_the_small_model(build, hr, redis):
    probe = CountingLLM(default="x [1].")
    await build(probe, cache=False).answer(ChatRequest(query=QUESTION), hr)
    prompt = estimate_messages_tokens(probe.messages[0])

    llm = CountingLLM(default="Short answer [1].")
    limit = 100_000
    pipeline = build(llm, cache=False, budget_limits={"acme_corp": limit}, on_exceeded="downgrade")
    # Room for prompt + 256 (qwen-turbo output) but not prompt + 1024 (qwen-plus output).
    await redis.set(budget_key("acme_corp"), limit - (prompt + 300))
    resp = await pipeline.answer(ChatRequest(query=QUESTION), hr)
    assert resp.model_used == "qwen-turbo" and resp.degraded
    assert llm.calls == ["qwen-turbo"]


async def test_budget_reject_policy_raises(build, hr, redis):
    llm = CountingLLM()
    pipeline = build(llm, cache=False, budget_limits={"acme_corp": 10})
    with pytest.raises(BudgetExceededError):
        await pipeline.answer(ChatRequest(query=QUESTION), hr)
    assert llm.calls == []
    assert await redis.get(budget_key("acme_corp")) in (None, b"0")


async def test_llm_failure_refunds_the_reservation(build, hr, redis):
    llm = CountingLLM(script=[NonRetryableLLMError("400", status_code=400)])
    pipeline = build(llm, cache=False, budget_limits={"acme_corp": 100_000})
    with pytest.raises(NonRetryableLLMError):
        await pipeline.answer(ChatRequest(query=QUESTION), hr)
    assert int(await redis.get(budget_key("acme_corp"))) == 0


# ---------- query rewrite ----------


async def test_multi_turn_query_is_rewritten_used_and_cacheable(build, hr):
    llm = CountingLLM(script=["What is the P6 salary band at Acme?"], default="Answer [1].")
    pipeline = build(llm, rewrite=True)
    req = ChatRequest(
        query="And what about P6?",
        history=[
            ChatTurn(role="user", content="What is the P5 salary band?"),
            ChatTurn(role="assistant", content="P5 is ... [1]"),
        ],
    )
    resp = await pipeline.answer(req, hr)
    assert llm.calls == ["qwen-turbo", "qwen-plus"]
    rewrite_prompt = llm.messages[0][1].content
    assert (
        "What is the P5 salary band?" in rewrite_prompt and "And what about P6?" in rewrite_prompt
    )
    answer_prompt = llm.messages[1][1].content
    assert "Question: What is the P6 salary band at Acme?" in answer_prompt
    assert not resp.cache_hit
    # Same self-contained question asked directly now hits the cached answer.
    again = await pipeline.answer(ChatRequest(query="What is the P6 salary band at Acme?"), hr)
    assert again.cache_hit


async def test_long_single_turn_query_is_not_rewritten(build, hr):
    llm = CountingLLM(default="Answer [1].")
    await build(llm, rewrite=True).answer(ChatRequest(query=QUESTION), hr)
    assert llm.calls == ["qwen-plus"]


async def test_short_query_is_rewritten(build, hr):
    llm = CountingLLM(script=["What is the P6 salary band?"], default="Answer [1].")
    await build(llm, rewrite=True).answer(ChatRequest(query="P6 band?"), hr)
    assert llm.calls == ["qwen-turbo", "qwen-plus"]


@pytest.mark.parametrize(
    "outcome",
    [NonRetryableLLMError("400", status_code=400), "   \n  ", "x" * 3000],
)
async def test_rewrite_failure_falls_back_to_raw_query(build, hr, outcome):
    llm = CountingLLM(script=[outcome], default="Answer [1].")
    req = ChatRequest(query="P6 band?")
    resp = await build(llm, rewrite=True).answer(req, hr)
    assert "Question: P6 band?" in llm.messages[-1][1].content
    assert resp.answer == "Answer [1]."


async def test_rewrite_timeout_falls_back_to_raw_query(build, hr):
    class SlowRewrite(CountingLLM):
        async def chat(self, messages, *, model, **kw):
            if model == "qwen-turbo":
                await asyncio.sleep(5)
            return await super().chat(messages, model=model, **kw)

    llm = SlowRewrite(default="Answer [1].")
    pipeline = build(llm, rewrite=True)
    pipeline._rewriter._timeout_s = 0.05
    resp = await pipeline.answer(ChatRequest(query="P6 band?"), hr)
    assert "Question: P6 band?" in llm.messages[-1][1].content and not resp.degraded


async def test_rewrite_skipped_when_budget_cannot_cover_it(build, hr, redis):
    llm = CountingLLM(default="Answer [1].")
    pipeline = build(llm, budget_limits={"acme_corp": 10**6}, rewrite=True)
    await redis.set(budget_key("acme_corp"), 10**6 - 100)  # not even room for a rewrite
    with pytest.raises(BudgetExceededError):
        await pipeline.answer(ChatRequest(query="P6 band?"), hr)
    assert llm.calls == []


# ---------- streaming pipeline + SSE route ----------


async def test_stream_events_and_cached_replay(build, hr):
    llm = CountingLLM(default="P6 is 8,300 USD [1].")
    pipeline = build(llm)
    events = [e async for e in pipeline.stream(ChatRequest(query=QUESTION, stream=True), hr)]
    kinds = [e.kind for e in events]
    assert kinds[0] == "meta" and kinds[-1] == "done" and "delta" in kinds
    assert events[0].data["model_used"] == "qwen-plus" and not events[0].data["degraded"]
    text = "".join(e.data["text"] for e in events if e.kind == "delta")
    assert text == "P6 is 8,300 USD [1]."
    done = events[-1].data
    assert done["answer"] == text and done["citations"][0]["index"] == 1

    replay = [e async for e in pipeline.stream(ChatRequest(query=QUESTION, stream=True), hr)]
    assert replay[0].data["cache_hit"] and replay[-1].data["cost_usd"] == 0
    assert len(llm.messages) == 1


async def test_stream_settles_budget_with_actual_usage(build, hr, redis):
    llm = CountingLLM(default="Answer [1].", usage=77)
    pipeline = build(llm, cache=False, budget_limits={"acme_corp": 100_000})
    [_ async for _ in pipeline.stream(ChatRequest(query=QUESTION, stream=True), hr)]
    assert int(await redis.get(budget_key("acme_corp"))) == 77


@pytest.fixture
def sse_client(settings, build):
    state: dict = {}

    def use(llm, **kw):
        state["pipeline"] = build(llm, **kw)

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_pipeline] = lambda: state["pipeline"]
    with TestClient(app) as client:
        yield client, use
    app.dependency_overrides.clear()


def _post(client, make_token, body: dict):
    token = make_token(tenant="acme_corp", role=Role.HR)
    return client.post("/v1/chat", json=body, headers={"Authorization": f"Bearer {token}"})


def test_sse_route_streams_meta_deltas_done(sse_client, make_token):
    client, use = sse_client
    use(CountingLLM(default="P6 is 8,300 USD [1]."))
    r = _post(client, make_token, {"query": QUESTION, "stream": True})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert [e[0] for e in events][0] == "meta" and events[-1][0] == "done"
    assert events[0][1]["trace_id"] == r.headers["X-Trace-ID"]
    assert events[-1][1]["answer"] == "P6 is 8,300 USD [1]."


def test_sse_failure_before_first_token_is_a_plain_http_error(sse_client, make_token):
    client, use = sse_client
    use(CountingLLM(script=[NonRetryableLLMError("401", status_code=401)]))
    r = _post(client, make_token, {"query": QUESTION, "stream": True})
    assert r.status_code == 502
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["trace_id"] == r.headers["X-Trace-ID"]


def test_sse_failure_after_first_token_sends_error_event(sse_client, make_token):
    class Breaks(CountingLLM):
        async def stream(self, messages, *, model, **kw):
            self.calls.append(model)
            yield StreamChunk(delta="P6 is ")
            raise RetryableLLMError("503", status_code=503)

    client, use = sse_client
    llm = Breaks()
    use(llm)
    r = _post(client, make_token, {"query": QUESTION, "stream": True})
    assert r.status_code == 200
    events = parse_sse(r.text)
    names = [e[0] for e in events]
    assert names == ["meta", "delta", "error"]
    assert events[-1][1]["trace_id"] == r.headers["X-Trace-ID"]
    assert llm.calls == ["qwen-plus"]  # no retry / switch once text was sent


def test_budget_exhausted_returns_429_with_retry_after(sse_client, make_token):
    client, use = sse_client
    use(CountingLLM(), cache=False, budget_limits={"acme_corp": 1})
    for stream in (False, True):
        r = _post(client, make_token, {"query": QUESTION, "stream": stream})
        assert r.status_code == 429
        assert 0 < int(r.headers["Retry-After"]) <= 86_401
        assert json.loads(r.text)["trace_id"] == r.headers["X-Trace-ID"]
