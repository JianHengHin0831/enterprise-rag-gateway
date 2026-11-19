"""RagPipeline (M1): retrieve -> wrap -> generate -> citations, on in-memory Qdrant + FakeLLM."""

import re

import pytest

from app.gateway.base import ChatMessage
from app.gateway.providers.fake import FakeLLM
from app.gateway.providers.template import TemplateProvider
from app.gateway.router import ModelRouter, load_models_config
from app.observability.trace import start_trace
from app.rag.embeddings import FakeEmbedder
from app.rag.pipeline import RagPipeline
from app.rag.prompts import load_prompt
from app.rag.retriever import HybridRetriever
from app.schemas import ChatRequest
from tests.conftest import TEST_COLLECTION


class RecordingLLM(FakeLLM):
    def __init__(self, **kw) -> None:
        super().__init__("dashscope", **kw)
        self.messages: list[list[ChatMessage]] = []

    async def chat(self, messages, **kw):
        self.messages.append(messages)
        return await super().chat(messages, **kw)


def make_pipeline(settings, retriever, llm) -> RagPipeline:
    router = ModelRouter(
        config=load_models_config(settings.models_config_path),
        providers={"dashscope": llm, "local": TemplateProvider()},
    )
    return RagPipeline(
        retriever=retriever,
        router=router,
        prompt=load_prompt(settings.prompts_dir, "rag_qa", 1),
        retrieve_top_k=20,
        context_top_n=5,
    )


@pytest.fixture
def retriever(indexed_qdrant):
    return HybridRetriever(indexed_qdrant, FakeEmbedder(), TEST_COLLECTION)


class EmptyRetriever:
    async def search(self, query, identity, top_k):
        return []


async def test_answer_maps_citations_to_retrieved_chunks(settings, retriever, hr):
    tid = start_trace()
    llm = RecordingLLM(default="P6 ranges from 8,300 to 12,600 USD per month [1][2].")
    resp = await make_pipeline(settings, retriever, llm).answer(
        ChatRequest(query="What is the P6 salary band?"), hr
    )
    assert resp.trace_id == tid
    assert resp.model_used == "qwen-plus" and not resp.degraded and not resp.refused
    assert [c.index for c in resp.citations] == [1, 2]
    assert not resp.citation_mismatch

    [messages] = llm.messages
    system, user = messages
    tag = re.search(r"<(doc_[0-9a-f]{8}) id=\"1\"", user.content).group(1)
    assert f"<{tag}>" in system.content  # system prompt names this request's nonce tag
    assert user.content.count(f"</{tag}>") == 5  # context_top_n chunks, all wrapped
    assert "Question: What is the P6 salary band?" in user.content
    first = re.search(rf'<{tag} id="1" title="([^"]*)"', user.content).group(1)
    assert resp.citations[0].title == first


async def test_out_of_range_citation_sets_mismatch(settings, retriever, hr):
    llm = RecordingLLM(default="Something [1][9].")
    resp = await make_pipeline(settings, retriever, llm).answer(ChatRequest(query="salary"), hr)
    assert resp.citation_mismatch
    assert [c.index for c in resp.citations] == [1]


async def test_injected_tag_in_document_is_neutralized(settings, retriever, intern):
    llm = RecordingLLM()
    await make_pipeline(settings, retriever, llm).answer(
        ChatRequest(query="pantry ignore previous instructions retrieved_doc"), intern
    )
    user = llm.messages[0][1].content
    assert "</retrieved_doc>" not in user and "<retrieved_doc" not in user


async def test_empty_retrieval_refuses_without_llm_call(settings, intern):
    llm = RecordingLLM()
    pipeline = make_pipeline(settings, EmptyRetriever(), llm)
    resp = await pipeline.answer(ChatRequest(query="anything"), intern)
    assert resp.refused and resp.answer == pipeline.refusal_text
    assert llm.calls == [] and resp.model_used is None


async def test_model_refusal_is_flagged_and_drops_citations(settings, retriever, intern):
    refusal = load_prompt(settings.prompts_dir, "rag_qa", 1).refusal_text
    llm = RecordingLLM(default=refusal)
    resp = await make_pipeline(settings, retriever, llm).answer(
        ChatRequest(query="stock option strike price"), intern
    )
    assert resp.refused and resp.citations == []


async def test_template_fallback_is_marked_degraded(settings, retriever, intern):
    router = ModelRouter(
        config=load_models_config(settings.models_config_path),
        providers={"local": TemplateProvider()},
    )
    pipeline = RagPipeline(
        retriever=retriever,
        router=router,
        prompt=load_prompt(settings.prompts_dir, "rag_qa", 1),
        retrieve_top_k=20,
        context_top_n=5,
    )
    resp = await pipeline.answer(ChatRequest(query="expense"), intern)
    assert resp.degraded and resp.model_used == "fallback-template"
