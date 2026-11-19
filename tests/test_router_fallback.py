"""Milestone acceptance tests. Remove the skip when implementing; do not weaken assertions."""

import pytest

from app.gateway.base import ChatMessage, RateLimitedError
from app.gateway.providers.fake import FakeLLM
from app.gateway.router import ModelRouter

CONFIG = {
    "models": {
        "primary": {"provider": "p1", "model": "big", "timeout_s": 5, "max_output_tokens": 64},
        "backup": {"provider": "p2", "model": "small", "timeout_s": 5, "max_output_tokens": 64},
    },
    "routes": {"answer": ["primary", "backup"], "rewrite": ["backup"]},
    "circuit_breaker": {"failure_threshold": 5, "recovery_timeout_s": 30, "half_open_max_calls": 2},
    "retry": {"max_attempts": 1, "base_delay_s": 0, "max_delay_s": 0, "request_deadline_s": 5},
}
MSGS = [ChatMessage(role="user", content="hi")]


async def test_breaker_opens_after_5_429s_and_6th_call_skips_primary():
    primary = FakeLLM("p1", script=[RateLimitedError("429", status_code=429)] * 10)
    backup = FakeLLM("p2", default="backup answer [1]")
    router = ModelRouter(config=CONFIG, providers={"p1": primary, "p2": backup})

    for _ in range(5):
        result = await router.chat("answer", MSGS)
        assert result.degraded and result.model_key == "backup"
    assert len(primary.calls) == 5

    result = await router.chat("answer", MSGS)
    assert len(primary.calls) == 5  # breaker open: primary not called
    assert result.degraded and result.response.text == "backup answer [1]"


async def test_non_retryable_error_does_not_trip_breaker():
    from app.gateway.base import NonRetryableLLMError

    primary = FakeLLM("p1", script=[NonRetryableLLMError("400", status_code=400)] * 10)
    router = ModelRouter(config=CONFIG, providers={"p1": primary, "p2": FakeLLM("p2")})
    for _ in range(6):
        with pytest.raises(NonRetryableLLMError):
            await router.chat("answer", MSGS)
    assert router.breaker("primary").state == "closed"


async def test_second_identical_question_hits_cache_with_zero_cost(settings, indexed_qdrant, hr):
    from fakeredis import FakeAsyncRedis

    from app.cache.layered import LayeredCache
    from app.gateway.providers.template import TemplateProvider
    from app.gateway.router import load_models_config
    from app.rag.embeddings import FakeEmbedder
    from app.rag.pipeline import RagPipeline
    from app.rag.prompts import load_prompt
    from app.rag.retriever import HybridRetriever
    from app.schemas import ChatRequest
    from tests.conftest import TEST_COLLECTION

    llm = FakeLLM("dashscope", default="P6 ranges from 8,300 to 12,600 USD per month [1].")
    pipeline = RagPipeline(
        retriever=HybridRetriever(indexed_qdrant, FakeEmbedder(), TEST_COLLECTION),
        router=ModelRouter(
            config=load_models_config(settings.models_config_path),
            providers={"dashscope": llm, "local": TemplateProvider()},
        ),
        prompt=load_prompt(settings.prompts_dir, "rag_qa", 1),
        retrieve_top_k=20,
        context_top_n=5,
        cache=LayeredCache(FakeAsyncRedis(), l1_size=500, l1_ttl_s=60, l2_ttl_s=3600),
    )
    question = ChatRequest(query="What is the P6 salary band?")

    first = await pipeline.answer(question, hr)
    second = await pipeline.answer(question, hr)

    assert first.cache_hit is False
    assert second.cache_hit is True and second.cost_usd == 0
    assert second.answer == first.answer
    assert llm.calls == ["qwen-plus"]  # the second answer made no LLM call
