"""M3: rerank Top-20 -> Top-5, low-score refusal gate (no LLM call), post-stream citation
check with disclaimer + metric. Real retrieval on in-memory Qdrant, FakeReranker, FakeLLM."""

import pytest
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient

from app.cache.keys import index_version_key
from app.cache.layered import LayeredCache
from app.gateway.base import ChatMessage
from app.gateway.providers.fake import FakeLLM
from app.gateway.providers.template import TemplateProvider
from app.gateway.router import ModelRouter
from app.observability import metrics
from app.observability.trace import current_spans, start_trace
from app.rag.citations import DISCLAIMER
from app.rag.embeddings import FakeEmbedder
from app.rag.pipeline import RagPipeline
from app.rag.prompts import load_prompt
from app.rag.reranker import FakeReranker, apply_scores, passes_gate
from app.rag.retriever import HybridRetriever
from app.schemas import ChatRequest, RetrievedChunk
from tests.conftest import TEST_COLLECTION, direct_vendor_models_config

ANSWERABLE = "What is the P6 salary band?"
NO_ANSWER = "What is the strike price of the company's stock options?"  # golden N-01
THRESHOLD = 0.5


class RecordingLLM(FakeLLM):
    def __init__(self, **kw) -> None:
        super().__init__("dashscope", **kw)
        self.messages: list[list[ChatMessage]] = []

    async def chat(self, messages, **kw):
        self.messages.append(messages)
        return await super().chat(messages, **kw)

    async def stream(self, messages, **kw):
        self.messages.append(messages)
        async for c in super().stream(messages, **kw):
            yield c


class RecordingReranker(FakeReranker):
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []  # (candidates, top_n)

    async def rerank(self, query, chunks, top_n):
        self.calls.append((len(chunks), top_n))
        return await super().rerank(query, chunks, top_n)


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
    def _build(llm, *, reranker=None, threshold=THRESHOLD, cache=False):
        router = ModelRouter(
            config=direct_vendor_models_config(settings.models_config_path),
            providers={"dashscope": llm, "local": TemplateProvider()},
        )
        return RagPipeline(
            retriever=HybridRetriever(indexed_qdrant, FakeEmbedder(), TEST_COLLECTION),
            router=router,
            prompt=load_prompt(settings.prompts_dir, "rag_qa", 1),
            retrieve_top_k=20,
            context_top_n=5,
            cache=LayeredCache(redis, l1_size=50, l1_ttl_s=60, l2_ttl_s=60) if cache else None,
            reranker=reranker if reranker is not None else FakeReranker(),
            rerank_threshold=threshold,
        )

    return _build


def _chunk(doc: str, text: str, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(doc_id=doc, chunk_index=0, text=text, score=score)


# ---------- reranker units ----------


def test_apply_scores_sorts_by_rerank_score_and_truncates():
    chunks = [_chunk("a", "x"), _chunk("b", "y"), _chunk("c", "z")]
    out = apply_scores(chunks, [0.1, 0.9, 0.5], top_n=2)
    assert [c.doc_id for c in out] == ["b", "c"]
    assert [c.rerank_score for c in out] == [0.9, 0.5]
    assert chunks[0].rerank_score is None  # inputs untouched


def test_apply_scores_rejects_a_score_count_mismatch():
    with pytest.raises(ValueError):
        apply_scores([_chunk("a", "x")], [0.1, 0.2], top_n=1)


async def test_fake_reranker_promotes_the_relevant_chunk():
    chunks = [_chunk("noise", "cafeteria menu"), _chunk("hit", "the P6 salary band is wide")]
    out = await FakeReranker().rerank(ANSWERABLE, chunks, top_n=5)
    assert out[0].doc_id == "hit" and out[0].rerank_score == 1.0
    assert out[1].rerank_score == 0.0


def test_gate_requires_a_scored_top_hit_at_or_above_threshold():
    scored = [_chunk("a", "x").model_copy(update={"rerank_score": 0.5})]
    assert passes_gate(scored, 0.5)
    assert not passes_gate(scored, 0.51)
    assert not passes_gate([], 0.0)
    assert not passes_gate([_chunk("a", "x")], 0.0)  # unscored never passes


# ---------- pipeline: rerank + gate ----------


async def test_reranks_top20_and_sends_top5_in_rerank_order(build, hr):
    llm = RecordingLLM(default="P6 is 8,300 to 12,600 USD [1].")
    reranker = RecordingReranker()
    start_trace()
    resp = await build(llm, reranker=reranker).answer(ChatRequest(query=ANSWERABLE), hr)
    assert not resp.refused and resp.citations
    candidates, top_n = reranker.calls[0]
    assert top_n == 5 and 5 < candidates <= 20
    rerank_span = next(s for s in current_spans() if s.name == "rerank")
    assert rerank_span.attrs["passed"] is True
    assert rerank_span.attrs["top_score"] >= THRESHOLD
    assert len(rerank_span.attrs["context_refs"]) == 5
    # The first reference in the prompt is the reranker's top chunk.
    first_ref = rerank_span.attrs["context_refs"][0].split("#")[0]
    assert resp.citations[0].doc_id == first_ref
    assert metrics.get("rag_rerank_gate_total", decision="passed") == 1


async def test_m3_acceptance_no_answer_is_refused_without_llm_span(build, hr):
    """ROADMAP M3 acceptance: a question with no answer in the corpus -> low-score refusal,
    no `llm` span in the trace."""
    llm = RecordingLLM()
    pipeline = build(llm)
    start_trace()
    resp = await pipeline.answer(ChatRequest(query=NO_ANSWER), hr)
    assert resp.refused and resp.answer == pipeline.refusal_text
    assert resp.cost_usd == 0 and resp.model_used is None and not resp.citations
    names = [s.name for s in current_spans()]
    assert "rerank" in names and "llm" not in names
    assert llm.calls == [] and llm.messages == []
    assert metrics.get("rag_refusals_total", reason="low_score") == 1
    assert metrics.get("rag_rerank_gate_total", decision="refused") == 1


async def test_low_score_refusal_streams_without_llm(build, hr):
    llm = RecordingLLM()
    start_trace()
    events = [e async for e in build(llm).stream(ChatRequest(query=NO_ANSWER), hr)]
    assert [e.kind for e in events] == ["meta", "delta", "done"]
    assert events[-1].data["refused"] is True
    assert llm.calls == []


async def test_threshold_zero_lets_the_same_question_through(build, hr):
    llm = RecordingLLM(default="No relevant information.")
    await build(llm, threshold=0.0).answer(ChatRequest(query=NO_ANSWER), hr)
    assert len(llm.calls) == 1


async def test_low_score_refusal_is_not_cached(build, hr, redis):
    await redis.set(index_version_key(hr.tenant_id), "1")
    llm = RecordingLLM()
    pipeline = build(llm, cache=True)
    start_trace()
    await pipeline.answer(ChatRequest(query=NO_ANSWER), hr)
    start_trace()
    second = await pipeline.answer(ChatRequest(query=NO_ANSWER), hr)
    assert second.refused and not second.cache_hit  # re-gated, so a new threshold applies
    assert metrics.get("rag_refusals_total", reason="low_score") == 2


# ---------- citation check + disclaimer ----------


async def test_citation_mismatch_appends_disclaimer_and_records_metric(build, hr, redis):
    await redis.set(index_version_key(hr.tenant_id), "1")
    llm = RecordingLLM(default="P6 is 8,300 to 12,600 USD [1][9].")
    pipeline = build(llm, cache=True)
    start_trace()
    resp = await pipeline.answer(ChatRequest(query=ANSWERABLE), hr)
    assert resp.citation_mismatch
    assert resp.answer == "P6 is 8,300 to 12,600 USD [1][9]." + DISCLAIMER
    assert [c.index for c in resp.citations] == [1]  # the invalid [9] is dropped
    assert metrics.get("rag_citation_mismatch_total") == 1
    start_trace()
    again = await pipeline.answer(ChatRequest(query=ANSWERABLE), hr)
    assert not again.cache_hit  # mismatched answers are never cached


async def test_stream_runs_citation_check_on_full_text_and_streams_disclaimer(build, hr):
    # "[9]" is split across 4-char FakeLLM deltas ("SD [", "9]."), so only a check on the
    # complete text can see it.
    llm = RecordingLLM(default="P6 is 8,300 USD [9].")
    start_trace()
    events = [e async for e in build(llm).stream(ChatRequest(query=ANSWERABLE), hr)]
    deltas = "".join(e.data["text"] for e in events if e.kind == "delta")
    done = events[-1]
    assert done.kind == "done" and done.data["citation_mismatch"] is True
    assert deltas.strip() == done.data["answer"]
    assert done.data["answer"].endswith(DISCLAIMER)
    assert events[-2].kind == "delta" and events[-2].data["text"] == DISCLAIMER
    assert metrics.get("rag_citation_mismatch_total") == 1


async def test_valid_citations_get_no_disclaimer(build, hr):
    llm = RecordingLLM(default="P6 is 8,300 USD [1].")
    start_trace()
    events = [e async for e in build(llm).stream(ChatRequest(query=ANSWERABLE), hr)]
    assert DISCLAIMER not in events[-1].data["answer"]
    assert not any(e.kind == "delta" and e.data["text"] == DISCLAIMER for e in events)
    assert metrics.get("rag_citation_mismatch_total") == 0


async def test_model_refusal_gets_no_disclaimer(build, hr, settings):
    refusal = load_prompt(settings.prompts_dir, "rag_qa", 1).refusal_text
    llm = RecordingLLM(default=f"{refusal} [7]")
    start_trace()
    resp = await build(llm).answer(ChatRequest(query=ANSWERABLE), hr)
    assert resp.refused and not resp.citation_mismatch and DISCLAIMER not in resp.answer
    assert metrics.get("rag_refusals_total", reason="model") == 1


# ---------- /metrics ----------


def test_metrics_endpoint_renders_counters(settings):
    from app.config import get_settings
    from app.main import app

    metrics.inc("rag_citation_mismatch_total")
    metrics.inc("rag_refusals_total", reason="low_score")
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        with TestClient(app) as client:
            body = client.get("/metrics").text
    finally:
        app.dependency_overrides.clear()
    assert "rag_citation_mismatch_total 1" in body
    assert 'rag_refusals_total{reason="low_score"} 1' in body


def test_unknown_metric_is_rejected():
    with pytest.raises(KeyError):
        metrics.inc("not_a_metric")


def test_build_reranker_device_overrides_embed_device():
    from app.config import Settings
    from app.rag.reranker import BgeReranker, build_reranker

    base = {"reranker": "bge-reranker", "embed_device": "cpu", "embed_use_fp16": True}
    inherited = build_reranker(Settings(**base))
    assert isinstance(inherited, BgeReranker)
    assert inherited._device == "cpu" and inherited._use_fp16 is True

    own = build_reranker(Settings(**base, rerank_device="cuda:0", rerank_use_fp16=False))
    assert own._device == "cuda:0" and own._use_fp16 is False
