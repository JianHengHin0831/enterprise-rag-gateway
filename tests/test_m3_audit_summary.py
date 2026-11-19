"""M3: append-only audit log, `rag_request_summary` event, optional LangFuse (OTLP) export.
Real retrieval on in-memory Qdrant + FakeReranker + FakeLLM; no network."""

import asyncio
import json
import logging

import anyio
import httpx
import pytest
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient

from app.cache.keys import index_version_key
from app.cache.layered import LayeredCache
from app.config import get_settings
from app.cost.budget import BudgetExceededError, BudgetGuard, budget_key
from app.gateway.base import NonRetryableLLMError, StreamChunk
from app.gateway.providers.fake import FakeLLM
from app.gateway.providers.template import TemplateProvider
from app.gateway.router import ModelRouter
from app.observability.langfuse import (
    OTLP_PATH,
    LangfuseExporter,
    build_exporter,
    build_otlp_payload,
)
from app.observability.trace import SpanRecord, current_spans, start_trace
from app.rag.embeddings import FakeEmbedder
from app.rag.pipeline import RagPipeline
from app.rag.prompts import load_prompt
from app.rag.reranker import FakeReranker
from app.rag.retriever import HybridRetriever
from app.schemas import ChatRequest, Identity, Role
from app.security.audit import AuditLogger, AuditRecord, Decision, build_record, hash_query
from tests.conftest import TEST_COLLECTION, direct_vendor_models_config

ANSWERABLE = "What is the P6 salary band?"
NO_ANSWER = "What is the strike price of the company's stock options?"
SUMMARY_FIELDS = {
    "decision",
    "steps_ms",
    "model_used",
    "degraded",
    "prompt_tokens",
    "completion_tokens",
    "estimated_prompt_tokens",
    "reserved_tokens",
    "actual_tokens",
    "prompt_estimate_error_pct",
    "cost_usd",
    "total_ms",
}


class RecordingExporter:
    def __init__(self) -> None:
        self.calls: list[tuple[dict, list[SpanRecord]]] = []

    def submit(self, summary, spans) -> None:
        self.calls.append((summary, spans))


@pytest.fixture
def redis():
    return FakeAsyncRedis()


@pytest.fixture
def audit(tmp_path) -> AuditLogger:
    return AuditLogger(tmp_path / "audit.jsonl")


@pytest.fixture
def build(settings, indexed_qdrant, redis, audit):
    def _build(llm, *, cache=False, budget=None, exporter=None, audit_logger=audit):
        cfg = direct_vendor_models_config(settings.models_config_path)
        cfg["retry"] = {**cfg["retry"], "base_delay_s": 0, "max_delay_s": 0}
        if budget is not None:
            cfg["budget"] = {"daily_tokens_per_tenant": budget, "on_exceeded": "reject"}
        router = ModelRouter(config=cfg, providers={"dashscope": llm, "local": TemplateProvider()})
        return RagPipeline(
            retriever=HybridRetriever(indexed_qdrant, FakeEmbedder(), TEST_COLLECTION),
            router=router,
            prompt=load_prompt(settings.prompts_dir, "rag_qa", 1),
            retrieve_top_k=20,
            context_top_n=5,
            budget=BudgetGuard.from_config(redis, cfg) if budget is not None else None,
            cache=LayeredCache(redis, l1_size=50, l1_ttl_s=60, l2_ttl_s=60) if cache else None,
            reranker=FakeReranker(),
            rerank_threshold=0.5,
            audit=audit_logger,
            exporter=exporter,
        )

    return _build


def _lines(audit: AuditLogger) -> list[AuditRecord]:
    if not audit.path.exists():
        return []
    text = audit.path.read_text(encoding="utf-8")
    return [AuditRecord.model_validate_json(line) for line in text.splitlines()]


def _summaries(caplog) -> list[dict]:
    return [r.rag_fields for r in caplog.records if r.getMessage() == "rag_request_summary"]


# ---------- audit: one line per request, every decision ----------


async def test_answer_is_audited_with_hash_docs_and_citations(build, hr, audit):
    tid = start_trace()
    resp = await build(FakeLLM("dashscope", default="P6 is 8,300 USD [1].")).answer(
        ChatRequest(query=ANSWERABLE), hr
    )
    [rec] = _lines(audit)
    assert rec.decision is Decision.ANSWERED and rec.trace_id == tid == resp.trace_id
    assert (rec.user_id, rec.tenant_id, rec.role) == ("u_hr", "acme_corp", "hr")
    assert rec.query_hash == hash_query(ANSWERABLE)
    assert rec.retrieved_doc_ids and rec.cited_doc_ids == [c.doc_id for c in resp.citations]
    assert rec.model_used == resp.model_used and rec.ts.tzinfo is not None
    assert ANSWERABLE not in audit.path.read_text(encoding="utf-8")  # never the raw query


async def test_low_score_refusal_is_audited(build, hr, audit):
    start_trace()
    await build(FakeLLM("dashscope")).answer(ChatRequest(query=NO_ANSWER), hr)
    [rec] = _lines(audit)
    assert rec.decision is Decision.REFUSED_LOW_SCORE and rec.retrieved_doc_ids
    assert rec.model_used is None


async def test_no_results_refusal_is_audited(settings, hr, audit):
    class Empty:
        async def search(self, query, identity, top_k):
            return []

    router = ModelRouter(
        config=direct_vendor_models_config(settings.models_config_path),
        providers={"dashscope": FakeLLM("dashscope"), "local": TemplateProvider()},
    )
    pipeline = RagPipeline(
        retriever=Empty(),
        router=router,
        prompt=load_prompt(settings.prompts_dir, "rag_qa", 1),
        retrieve_top_k=20,
        context_top_n=5,
        audit=audit,
    )
    start_trace()
    await pipeline.answer(ChatRequest(query="anything at all here"), hr)
    [rec] = _lines(audit)
    assert rec.decision is Decision.REFUSED_NO_RESULTS and rec.retrieved_doc_ids == []


async def test_model_refusal_is_audited(build, hr, audit, settings):
    refusal = load_prompt(settings.prompts_dir, "rag_qa", 1).refusal_text
    start_trace()
    await build(FakeLLM("dashscope", default=refusal)).answer(ChatRequest(query=ANSWERABLE), hr)
    assert _lines(audit)[0].decision is Decision.REFUSED_BY_MODEL


async def test_cache_hit_is_audited_with_cited_docs(build, hr, audit, redis):
    await redis.set(index_version_key(hr.tenant_id), "1")
    pipeline = build(FakeLLM("dashscope", default="P6 is 8,300 USD [1]."), cache=True)
    start_trace()
    await pipeline.answer(ChatRequest(query=ANSWERABLE), hr)
    start_trace()
    await pipeline.answer(ChatRequest(query=ANSWERABLE), hr)
    first, second = _lines(audit)
    assert (first.decision, second.decision) == (Decision.ANSWERED, Decision.CACHE_HIT)
    assert second.cited_doc_ids == first.cited_doc_ids and second.retrieved_doc_ids == []


async def test_budget_rejection_is_audited(build, hr, audit):
    start_trace()
    with pytest.raises(BudgetExceededError):
        await build(FakeLLM("dashscope"), budget={"acme_corp": 10}).answer(
            ChatRequest(query=ANSWERABLE), hr
        )
    [rec] = _lines(audit)
    assert rec.decision is Decision.BUDGET_EXCEEDED and rec.error_type is None


async def test_llm_error_is_audited_with_error_type(build, hr, audit):
    llm = FakeLLM("dashscope", script=[NonRetryableLLMError("bad request", status_code=400)])
    start_trace()
    with pytest.raises(NonRetryableLLMError):
        await build(llm).answer(ChatRequest(query=ANSWERABLE), hr)
    [rec] = _lines(audit)
    assert rec.decision is Decision.ERROR and rec.error_type == "NonRetryableLLMError"


async def test_stream_answer_is_audited_once(build, hr, audit):
    start_trace()
    events = [
        e
        async for e in build(FakeLLM("dashscope", default="P6 is 8,300 USD [1].")).stream(
            ChatRequest(query=ANSWERABLE, stream=True), hr
        )
    ]
    assert events[-1].kind == "done"
    [rec] = _lines(audit)
    assert rec.decision is Decision.ANSWERED and rec.cited_doc_ids


async def test_client_disconnect_mid_stream_is_audited(build, hr, audit):
    start_trace()
    gen = build(FakeLLM("dashscope", default="A long answer about P6 bands [1].")).stream(
        ChatRequest(query=ANSWERABLE, stream=True), hr
    )
    assert (await anext(gen)).kind == "meta"
    await anext(gen)  # one delta, then the client goes away
    await gen.aclose()
    [rec] = _lines(audit)
    assert rec.decision is Decision.ERROR and rec.error_type == "ClientDisconnected"


async def test_audit_is_append_only(build, hr, audit):
    pipeline = build(FakeLLM("dashscope", default="P6 is 8,300 USD [1]."))
    start_trace()
    await pipeline.answer(ChatRequest(query=ANSWERABLE), hr)
    first_bytes = audit.path.read_bytes()
    start_trace()
    await pipeline.answer(ChatRequest(query=NO_ANSWER), hr)
    after = audit.path.read_bytes()
    assert after.startswith(first_bytes) and len(_lines(audit)) == 2


async def test_concurrent_writes_never_interleave(audit):
    ident = Identity(tenant_id="acme_corp", role=Role.HR, user_id="u")
    records = [
        build_record(
            trace_id=f"t{i}",
            identity=ident,
            query=f"q{i}",
            retrieved_doc_ids=["d"] * 50,
            decision=Decision.ANSWERED,
        )
        for i in range(40)
    ]
    await asyncio.gather(*(audit.awrite(r) for r in records))
    assert sorted(r.trace_id for r in _lines(audit)) == sorted(f"t{i}" for i in range(40))


async def test_audit_failure_does_not_fail_the_request(build, hr):
    class Broken(AuditLogger):
        def __init__(self) -> None:
            pass

        async def awrite(self, record) -> None:
            raise OSError("disk full")

    start_trace()
    resp = await build(FakeLLM("dashscope", default="P6 [1]."), audit_logger=Broken()).answer(
        ChatRequest(query=ANSWERABLE), hr
    )
    assert not resp.refused


# ---------- rag_request_summary ----------


async def test_summary_event_has_the_roadmap_fields(build, hr, caplog):
    caplog.set_level(logging.INFO, logger="rag.pipeline")
    start_trace()
    resp = await build(FakeLLM("dashscope", default="P6 is 8,300 USD [1]."), budget={}).answer(
        ChatRequest(query=ANSWERABLE), hr
    )
    [summary] = _summaries(caplog)
    assert summary.keys() >= SUMMARY_FIELDS
    assert summary["decision"] == "answered" and summary["model_used"] == resp.model_used
    assert set(summary["steps_ms"]) >= {"retrieve", "rerank", "budget", "llm"}
    assert summary["prompt_tokens"] > 0 and summary["completion_tokens"] > 0
    assert summary["actual_tokens"] == summary["prompt_tokens"] + summary["completion_tokens"]
    assert summary["estimated_prompt_tokens"] > 0
    assert summary["reserved_tokens"] > summary["estimated_prompt_tokens"]  # + max output
    assert summary["total_ms"] >= max(summary["steps_ms"].values())
    assert ANSWERABLE not in json.dumps(summary)  # query text never in the summary


async def test_summary_for_refusal_has_no_llm_numbers(build, hr, caplog):
    caplog.set_level(logging.INFO, logger="rag.pipeline")
    start_trace()
    await build(FakeLLM("dashscope")).answer(ChatRequest(query=NO_ANSWER), hr)
    [summary] = _summaries(caplog)
    assert summary["decision"] == "refused_low_score" and summary["cost_usd"] == 0
    assert summary["prompt_tokens"] is None and "llm" not in summary["steps_ms"]


async def test_exactly_one_summary_per_request(build, hr, caplog):
    caplog.set_level(logging.INFO, logger="rag.pipeline")
    pipeline = build(FakeLLM("dashscope", default="P6 [1]."))
    for q in (ANSWERABLE, NO_ANSWER):
        start_trace()
        await pipeline.answer(ChatRequest(query=q), hr)
    assert [s["decision"] for s in _summaries(caplog)] == ["answered", "refused_low_score"]


# ---------- LangFuse (OTLP) export ----------


async def test_exporter_receives_summary_and_spans(build, hr):
    exporter = RecordingExporter()
    start_trace()
    await build(FakeLLM("dashscope", default="P6 [1]."), exporter=exporter).answer(
        ChatRequest(query=ANSWERABLE), hr
    )
    [(summary, spans)] = exporter.calls
    assert summary["decision"] == "answered"
    assert [s.name for s in spans] == [s.name for s in current_spans()]


def test_otlp_payload_shape_and_privacy():
    spans = [
        SpanRecord(name="retrieve", start_ts=1000.0, latency_ms=20, attrs={"hits": 3}),
        SpanRecord(
            name="llm",
            start_ts=1000.1,
            latency_ms=900,
            attrs={"model_used": "claude-sonnet", "prompt_tokens": 120, "completion_tokens": 30},
        ),
    ]
    summary = {
        "tenant_id": "acme_corp",
        "user_id": "u_hr",
        "decision": "answered",
        "cost_usd": 0.002,
        "error_type": None,
        "total_ms": 1000.0,
    }
    payload = build_otlp_payload("a" * 32, summary, spans)
    otlp = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
    root, retrieve, llm = otlp
    assert root["name"] == "rag_request" and "parentSpanId" not in root
    assert {s["traceId"] for s in otlp} == {"a" * 32}
    assert retrieve["parentSpanId"] == llm["parentSpanId"] == root["spanId"]
    assert all(len(s["spanId"]) == 16 for s in otlp)
    root_attrs = {a["key"]: a["value"] for a in root["attributes"]}
    assert root_attrs["langfuse.user.id"] == {"stringValue": "u_hr"}
    assert root_attrs["langfuse.trace.metadata.decision"] == {"stringValue": "answered"}
    llm_attrs = {a["key"]: a["value"] for a in llm["attributes"]}
    assert llm_attrs["langfuse.observation.type"] == {"stringValue": "generation"}
    assert llm_attrs["gen_ai.usage.input_tokens"] == {"intValue": "120"}
    assert llm_attrs["gen_ai.request.model"] == {"stringValue": "claude-sonnet"}
    assert int(llm["endTimeUnixNano"]) - int(llm["startTimeUnixNano"]) == 900_000_000


async def test_langfuse_exporter_posts_otlp_with_auth():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        exporter = LangfuseExporter(
            http, host="https://lf.example.test/", public_key="pk-lf-x", secret_key="sk-lf-y"
        )
        start_trace()
        exporter.submit({"decision": "answered", "total_ms": 5.0}, [])
        await exporter.aclose()
    [req] = seen
    assert str(req.url) == "https://lf.example.test" + OTLP_PATH
    assert req.headers["authorization"].startswith("Basic ")
    assert req.headers["x-langfuse-ingestion-version"] == "4"
    assert json.loads(req.content)["resourceSpans"]


async def test_langfuse_export_failure_is_swallowed():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        exporter = LangfuseExporter(
            http, host="https://lf.example.test", public_key="p", secret_key="s"
        )
        start_trace()
        exporter.submit({"total_ms": 1.0}, [])
        await exporter.aclose()  # no exception


def test_exporter_is_off_without_both_keys(settings):
    assert build_exporter(settings, httpx.AsyncClient()) is None
    half = settings.model_copy(update={"langfuse_public_key": "pk"})
    assert build_exporter(half, httpx.AsyncClient()) is None


# ---------- HTTP route writes the audit log configured in settings ----------


def test_chat_route_writes_audit_line(settings, make_token):
    from app.main import app

    # Nothing listens on port 9: retrieval fails (503), and that failure must still be audited.
    settings = settings.model_copy(update={"qdrant_url": "http://127.0.0.1:9"})
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat",
                headers={"Authorization": f"Bearer {make_token(role=Role.HR, user='u_route')}"},
                json={"query": ANSWERABLE},
            )
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 503
    [rec] = [
        AuditRecord.model_validate_json(line)
        for line in settings.audit_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rec.user_id == "u_route" and rec.decision is Decision.ERROR
    assert rec.trace_id == r.json()["trace_id"]


# ---------- client disconnect: cleanup must survive anyio cancellation ----------
#
# uvicorn reports ASGI spec 2.3, so Starlette's StreamingResponse runs `listen_for_disconnect`
# and cancels the response task group when the client goes away. anyio cancellation is
# level-triggered: every later await in that task is cancelled again. These tests drive
# `PipelineStreamingResponse` as a real ASGI app through both disconnect timings.


class SlowStreamLLM(FakeLLM):
    """First delta at once, then stalls as if the model were still generating."""

    async def stream(self, messages, **kw):
        self.calls.append(kw.get("model"))
        yield StreamChunk(delta="P6 is ")
        await asyncio.sleep(30)
        yield StreamChunk(delta="8,300 USD [1].")


async def _serve(
    pipeline, identity, *, disconnect_after_body_frames: int, backpressured: bool = False
) -> list[dict]:
    """Run /v1/chat's streaming response over ASGI; the client disconnects once it has seen
    `disconnect_after_body_frames` body frames.

    `backpressured=True` models our `@app.middleware("http")` (BaseHTTPMiddleware): it hands
    messages through a zero-buffer memory stream, so `send` blocks until the outer side reads,
    and on a disconnect the outer side cancels everything while the response is stuck in
    `send`. The body iterator is then suspended at a yield (or never started), so nothing in
    it can run the cleanup."""
    from app.main import PipelineStreamingResponse, _sse

    events = pipeline.stream(ChatRequest(query=ANSWERABLE, stream=True), identity)
    first = await anext(events)  # the endpoint awaits the first event before responding
    response = PipelineStreamingResponse(
        _sse(first, events), events=events, media_type="text/event-stream"
    )
    gone = asyncio.Event()
    sent: list[dict] = []

    async def receive() -> dict:
        await gone.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        if backpressured:
            gone.set()  # the client is gone; this write never completes
            await anyio.sleep_forever()
        sent.append(message)
        bodies = sum(m["type"] == "http.response.body" for m in sent)
        if bodies >= disconnect_after_body_frames > 0:
            gone.set()

    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}}
    await response(scope, receive, send)
    return sent


async def test_disconnect_before_body_starts_still_audits_and_settles(
    build, hr, audit, redis, caplog
):
    """The live failure (2026-11-20): the client left while rerank ran; the response was
    cancelled while blocked in `send`, so only the response-level close runs the cleanup."""
    pipeline = build(SlowStreamLLM("dashscope"), budget={"acme_corp": 1_000_000})
    start_trace()
    caplog.set_level(logging.INFO)
    sent = await _serve(pipeline, hr, disconnect_after_body_frames=0, backpressured=True)
    assert sent == []  # not a single byte reached the client
    # The inner generator (suspended inside the `llm` span) is closed before the summary,
    # not left to the garbage collector: the llm span and the settle precede the record.
    events = [r.getMessage() for r in caplog.records]
    llm_span = next(
        i for i, r in enumerate(caplog.records) if getattr(r, "rag_fields", {}).get("span") == "llm"
    )
    assert llm_span < events.index("rag_request_summary")
    [summary] = _summaries(caplog)
    assert "llm" in summary["steps_ms"]
    [rec] = _lines(audit)
    assert rec.decision is Decision.ERROR and rec.error_type == "ClientDisconnected"
    assert int(await redis.get(budget_key(hr.tenant_id))) > 0  # settled once, estimate kept


async def test_disconnect_while_generating_still_audits_and_settles(build, hr, audit, redis):
    """The client leaves after the first delta while the model is still generating: the
    cancellation lands inside the pipeline generator, so its own cleanup must be shielded."""
    pipeline = build(SlowStreamLLM("dashscope"), budget={"acme_corp": 1_000_000})
    start_trace()
    sent = await _serve(pipeline, hr, disconnect_after_body_frames=2)  # meta + one delta
    assert any(b"event: delta" in m.get("body", b"") for m in sent)
    [rec] = _lines(audit)
    assert rec.decision is Decision.ERROR and rec.error_type == "ClientDisconnected"
    assert rec.model_used == "qwen-plus"  # who produced the partial answer
    assert int(await redis.get(budget_key(hr.tenant_id))) > 0


async def test_completed_stream_is_audited_once_through_the_response(build, hr, audit):
    pipeline = build(FakeLLM("dashscope", default="P6 is 8,300 USD [1]."))
    start_trace()
    sent = await _serve(pipeline, hr, disconnect_after_body_frames=10_000)
    assert any(b"event: done" in m.get("body", b"") for m in sent)
    [rec] = _lines(audit)
    assert rec.decision is Decision.ANSWERED
