"""Truncated answers: the model stopped at its output-token limit (finish_reason "length" on
OpenAI-compatible / OpenRouter, "max_tokens" on Anthropic). Seen live on golden D-08: the
answer ended mid-sentence and was served and cached as if complete.

Invariant: a truncated answer is flagged (`truncated=true`), ends with TRUNCATION_NOTICE
(streams: one extra delta, so the client text matches `done`), is counted in
`rag_truncated_answers_total`, and is never cached."""

import pytest
from fakeredis import FakeAsyncRedis

from app.cache.keys import index_version_key, should_cache
from app.cache.layered import LayeredCache
from app.gateway.base import is_truncated
from app.gateway.providers.fake import FakeLLM
from app.gateway.providers.template import TemplateProvider
from app.gateway.router import ModelRouter
from app.observability import metrics
from app.observability.trace import start_trace
from app.rag.citations import DISCLAIMER, TRUNCATION_NOTICE
from app.rag.embeddings import FakeEmbedder
from app.rag.pipeline import RagPipeline
from app.rag.prompts import load_prompt
from app.rag.reranker import FakeReranker
from app.rag.retriever import HybridRetriever
from app.schemas import ChatRequest, ChatResponse
from tests.conftest import TEST_COLLECTION, direct_vendor_models_config

QUESTION = "What is the P6 salary band?"
CUT = "The P6 band is 8,300 to 12,600 USD [1]. Severance follows N+1, where N is the total"


@pytest.fixture(autouse=True)
def _reset_metrics():
    metrics.reset()
    yield
    metrics.reset()


@pytest.fixture
def redis():
    return FakeAsyncRedis()


@pytest.fixture
def build(settings, indexed_qdrant, redis):
    async def _build(llm: FakeLLM, *, cache: bool = False) -> RagPipeline:
        if cache:
            await redis.set(index_version_key("acme_corp"), 1)
        return RagPipeline(
            retriever=HybridRetriever(indexed_qdrant, FakeEmbedder(), TEST_COLLECTION),
            router=ModelRouter(
                config=direct_vendor_models_config(settings.models_config_path),
                providers={"dashscope": llm, "local": TemplateProvider()},
            ),
            prompt=load_prompt(settings.prompts_dir, "rag_qa", 1),
            retrieve_top_k=20,
            context_top_n=5,
            reranker=FakeReranker(),
            rerank_threshold=0.0,
            cache=LayeredCache(redis, l1_size=50, l1_ttl_s=60, l2_ttl_s=60) if cache else None,
        )

    return _build


@pytest.mark.parametrize(
    ("reason", "expected"),
    [("length", True), ("max_tokens", True), ("stop", False), ("end_turn", False), (None, False)],
)
def test_is_truncated(reason, expected):
    assert is_truncated(reason) is expected


def test_truncated_answers_are_never_cached():
    resp = ChatResponse(trace_id="t", answer="a [1]", model_used="claude-sonnet", truncated=True)
    assert not should_cache(resp, fallback_model_keys=frozenset({"fallback-template"}))


async def test_truncated_answer_is_flagged_with_notice_and_metric(build, hr):
    pipeline = await build(FakeLLM("dashscope", default=CUT, finish_reason="length"))
    start_trace()
    resp = await pipeline.answer(ChatRequest(query=QUESTION), hr)
    assert resp.truncated and not resp.refused
    assert resp.answer == CUT + TRUNCATION_NOTICE
    assert resp.citations  # the part before the cut is still cited
    assert metrics.get("rag_truncated_answers_total") == 1


async def test_complete_answer_is_not_flagged(build, hr):
    pipeline = await build(FakeLLM("dashscope", default="P6 is 8,300 to 12,600 USD [1]."))
    start_trace()
    resp = await pipeline.answer(ChatRequest(query=QUESTION), hr)
    assert not resp.truncated and TRUNCATION_NOTICE not in resp.answer
    assert metrics.get("rag_truncated_answers_total") == 0


async def test_truncated_answer_is_not_served_from_cache(build, hr):
    llm = FakeLLM("dashscope", default=CUT, finish_reason="length")
    pipeline = await build(llm, cache=True)
    for _ in range(2):
        start_trace()
        resp = await pipeline.answer(ChatRequest(query=QUESTION), hr)
        assert resp.truncated and not resp.cache_hit
    assert len(llm.calls) == 2  # asked the model again instead of replaying the cut answer


async def test_truncated_stream_ends_with_the_notice_and_matches_done(build, hr):
    pipeline = await build(FakeLLM("dashscope", default=CUT, finish_reason="length"))
    start_trace()
    events = [e async for e in pipeline.stream(ChatRequest(query=QUESTION, stream=True), hr)]
    deltas = [e.data["text"] for e in events if e.kind == "delta"]
    done = events[-1]
    assert done.kind == "done" and done.data["truncated"] is True
    assert deltas[-1] == TRUNCATION_NOTICE
    assert "".join(deltas) == done.data["answer"]


async def test_stream_with_bad_citation_and_truncation_sends_both_notes_in_order(build, hr):
    text = "See the policy [9]. Severance follows N+1, where N is"
    pipeline = await build(FakeLLM("dashscope", default=text, finish_reason="max_tokens"))
    start_trace()
    events = [e async for e in pipeline.stream(ChatRequest(query=QUESTION, stream=True), hr)]
    deltas = [e.data["text"] for e in events if e.kind == "delta"]
    done = events[-1].data
    assert done["citation_mismatch"] and done["truncated"]
    assert deltas[-2:] == [DISCLAIMER, TRUNCATION_NOTICE]
    assert "".join(deltas) == done["answer"]


async def test_a_truncated_refusal_is_just_a_refusal(build, hr, settings):
    refusal = load_prompt(settings.prompts_dir, "rag_qa", 1).refusal_text
    pipeline = await build(FakeLLM("dashscope", default=refusal, finish_reason="length"))
    start_trace()
    resp = await pipeline.answer(ChatRequest(query=QUESTION), hr)
    assert resp.refused and not resp.truncated
    assert TRUNCATION_NOTICE not in resp.answer
