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


@pytest.mark.skip(reason="M2: ModelRouter.chat not implemented")
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


@pytest.mark.skip(reason="M2: ModelRouter.chat not implemented")
async def test_non_retryable_error_does_not_trip_breaker():
    from app.gateway.base import NonRetryableLLMError

    primary = FakeLLM("p1", script=[NonRetryableLLMError("400", status_code=400)] * 10)
    router = ModelRouter(config=CONFIG, providers={"p1": primary, "p2": FakeLLM("p2")})
    for _ in range(6):
        with pytest.raises(NonRetryableLLMError):
            await router.chat("answer", MSGS)
    assert router.breaker("primary").state == "closed"


@pytest.mark.skip(reason="M2: LayeredCache not implemented")
async def test_second_identical_question_hits_cache_with_zero_cost(): ...
