"""ModelRouter retry / backoff / deadline / breaker isolation (M2), with a fake clock."""

import asyncio
import random

import pytest

from app.gateway.base import (
    AllModelsFailedError,
    ChatMessage,
    ContentRefusalError,
    LLMResponse,
    NonRetryableLLMError,
    RateLimitedError,
    RetryableLLMError,
    Usage,
)
from app.gateway.circuit_breaker import BreakerState
from app.gateway.providers.fake import FakeLLM
from app.gateway.providers.template import TemplateProvider
from app.gateway.router import ModelRouter, RetryPolicy

MSGS = [ChatMessage(role="user", content="hi")]


def _config(**retry) -> dict:
    return {
        "models": {
            "primary": {"provider": "p1", "model": "big", "timeout_s": 5, "max_output_tokens": 64},
            # Same upstream model under a second key: must share the primary's breaker.
            "primary-alias": {
                "provider": "p1",
                "model": "big",
                "timeout_s": 5,
                "max_output_tokens": 32,
            },
            "backup": {"provider": "p2", "model": "small", "timeout_s": 5, "max_output_tokens": 64},
            "template": {
                "provider": "local",
                "model": "tpl",
                "timeout_s": 1,
                "max_output_tokens": 0,
            },
        },
        "routes": {"answer": ["primary", "backup", "template"], "rewrite": ["primary-alias"]},
        "circuit_breaker": {
            "failure_threshold": 5,
            "recovery_timeout_s": 30,
            "half_open_max_calls": 1,
        },
        "retry": {
            "max_attempts": 3,
            "base_delay_s": 0.5,
            "max_delay_s": 4.0,
            "request_deadline_s": 15,
            **retry,
        },
    }


class FakeTime:
    """Monotonic clock + sleep that advances it instantly."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class MaxRandom(random.Random):
    """`uniform(a, b)` always returns b, so jittered delays hit their cap."""

    def uniform(self, a: float, b: float) -> float:
        return b


def _router(providers, t: FakeTime, rng: random.Random | None = None, **retry) -> ModelRouter:
    return ModelRouter(
        config=_config(**retry), providers=providers, clock=t, sleep=t.sleep, rng=rng
    )


def _err503() -> RetryableLLMError:
    return RetryableLLMError("503", status_code=503)


async def test_retry_then_success_on_primary_is_not_degraded():
    t = FakeTime()
    p1 = FakeLLM("p1", script=[_err503(), _err503()], default="primary [1]")
    router = _router({"p1": p1, "p2": FakeLLM("p2")}, t, rng=MaxRandom())
    result = await router.chat("answer", MSGS)
    assert result.model_key == "primary" and not result.degraded
    assert result.attempts == 3 and p1.calls == ["big"] * 3
    assert t.sleeps == [0.5, 1.0]  # exponential: base * 2**attempt (full jitter at its cap)


def test_full_jitter_stays_within_exponential_cap():
    router = ModelRouter(config=_config(), providers={}, rng=random.Random(7))
    for attempt, cap in [(0, 0.5), (1, 1.0), (2, 2.0), (3, 4.0), (6, 4.0)]:
        delays = [router.backoff_s(attempt, None) for _ in range(200)]
        assert all(0.0 <= d <= cap for d in delays)
        assert max(delays) > cap * 0.8 and min(delays) < cap * 0.2  # spread over [0, cap]


async def test_retry_after_is_honoured():
    t = FakeTime()
    p1 = FakeLLM("p1", script=[RateLimitedError("429", status_code=429, retry_after_s=3.0)])
    router = _router({"p1": p1, "p2": FakeLLM("p2")}, t, rng=random.Random(1))
    result = await router.chat("answer", MSGS)
    assert result.model_key == "primary"
    assert t.sleeps == [3.0]  # jitter (<= 0.5) never shortens the server's hint


async def test_retry_after_longer_than_max_delay_falls_back_without_waiting():
    t = FakeTime()
    p1 = FakeLLM("p1", script=[RateLimitedError("429", status_code=429, retry_after_s=60)])
    p2 = FakeLLM("p2", default="backup [1]")
    router = _router({"p1": p1, "p2": p2}, t)
    result = await router.chat("answer", MSGS)
    assert t.sleeps == [] and len(p1.calls) == 1
    assert result.model_key == "backup" and result.degraded


async def test_exhausted_retries_fall_back_down_the_chain():
    t = FakeTime()
    p1 = FakeLLM("p1", script=[_err503()] * 3)
    p2 = FakeLLM("p2", script=[_err503()] * 3)
    router = _router({"p1": p1, "p2": p2, "local": TemplateProvider()}, t)
    result = await router.chat("answer", MSGS)
    assert len(p1.calls) == 3 and len(p2.calls) == 3
    assert result.model_key == "template" and result.degraded and result.attempts == 7


async def test_all_models_failing_raises_all_models_failed_with_last_error():
    t = FakeTime()
    last = RetryableLLMError("502", status_code=502)
    p1 = FakeLLM("p1", script=[_err503()] * 3)
    p2 = FakeLLM("p2", script=[_err503(), _err503(), last])
    router = _router({"p1": p1, "p2": p2}, t)  # no template configured
    with pytest.raises(AllModelsFailedError) as info:
        await router.chat("answer", MSGS)
    assert info.value.last_error is last and info.value.retryable


@pytest.mark.parametrize(
    "error",
    [
        NonRetryableLLMError("400", status_code=400),
        NonRetryableLLMError("401", status_code=401),
        NonRetryableLLMError("403", status_code=403),
        ContentRefusalError("refused"),
    ],
)
async def test_non_retryable_error_is_raised_without_retry_or_fallback(error):
    t = FakeTime()
    p1 = FakeLLM("p1", script=[error])
    p2 = FakeLLM("p2")
    router = _router({"p1": p1, "p2": p2, "local": TemplateProvider()}, t)
    with pytest.raises(type(error)):
        await router.chat("answer", MSGS)
    assert len(p1.calls) == 1 and p2.calls == [] and t.sleeps == []
    assert router.breaker("primary").state is BreakerState.CLOSED


async def test_backoff_never_sleeps_past_the_deadline():
    t = FakeTime()
    p1 = FakeLLM("p1", script=[_err503()] * 3)
    p2 = FakeLLM("p2", default="backup [1]")
    # Deadline 1s: first backoff (0.5) fits, the second (1.0) would cross it.
    router = _router({"p1": p1, "p2": p2}, t, rng=MaxRandom(), request_deadline_s=1.0)
    result = await router.chat("answer", MSGS)
    assert t.sleeps == [0.5] and len(p1.calls) == 2
    assert result.model_key == "backup"


async def test_deadline_skips_remote_models_but_template_still_answers():
    t = FakeTime()

    class SlowThenFail(FakeLLM):
        async def chat(self, messages, *, model, **kw):
            self.calls.append(model)
            t.now += 20  # this call alone burns the whole deadline
            raise _err503()

    p1 = SlowThenFail("p1")
    p2 = FakeLLM("p2")
    router = _router({"p1": p1, "p2": p2, "local": TemplateProvider()}, t)
    result = await router.chat("answer", MSGS)
    assert len(p1.calls) == 1 and t.sleeps == []  # no retry once the deadline is gone
    assert p2.calls == []
    assert result.model_key == "template" and result.degraded


async def test_hung_provider_is_cut_off_by_the_wall_clock_deadline():
    class Hang(FakeLLM):
        async def chat(self, messages, **kw) -> LLMResponse:
            await asyncio.sleep(10)
            raise AssertionError("unreachable")

    p2 = FakeLLM("p2", default="backup [1]")
    router = ModelRouter(
        config=_config(max_attempts=1, request_deadline_s=0.05),
        providers={"p1": Hang("p1"), "p2": p2, "local": TemplateProvider()},
    )
    loop = asyncio.get_running_loop()
    started = loop.time()
    result = await router.chat("answer", MSGS)
    assert loop.time() - started < 1.0  # not the provider's 10 s
    # Coarse OS timers may leave the backup a sliver of budget, so either fallback is fine.
    assert result.model_key in {"backup", "template"} and result.degraded
    assert router.breaker("primary")._consecutive_failures == 1  # timeout counts


async def test_open_breaker_skips_model_and_recovers_after_cooldown():
    t = FakeTime()
    p1 = FakeLLM("p1", script=[_err503()] * 5, default="primary again [1]")
    p2 = FakeLLM("p2", default="backup [1]")
    router = _router({"p1": p1, "p2": p2}, t, max_attempts=5, request_deadline_s=1000)
    result = await router.chat("answer", MSGS)
    assert len(p1.calls) == 5 and result.model_key == "backup"
    assert router.breaker("primary").state is BreakerState.OPEN

    await router.chat("answer", MSGS)
    assert len(p1.calls) == 5  # skipped while open

    t.now += 31  # past recovery_timeout_s -> half-open probe succeeds
    result = await router.chat("answer", MSGS)
    assert result.model_key == "primary" and not result.degraded
    assert router.breaker("primary").state is BreakerState.CLOSED


async def test_non_retryable_error_in_half_open_releases_the_probe_slot():
    t = FakeTime()
    p1 = FakeLLM(
        "p1",
        script=[_err503()] * 5 + [NonRetryableLLMError("400", status_code=400)],
        default="primary [1]",
    )
    router = _router({"p1": p1, "p2": FakeLLM("p2")}, t, max_attempts=5, request_deadline_s=1000)
    await router.chat("answer", MSGS)  # trips the breaker
    t.now += 31
    with pytest.raises(NonRetryableLLMError):
        await router.chat("answer", MSGS)  # the single half-open probe hits a 400
    assert router.breaker("primary").state is BreakerState.HALF_OPEN
    result = await router.chat("answer", MSGS)  # slot was released: a new probe is allowed
    assert result.model_key == "primary"


async def test_breakers_are_per_provider_model():
    t = FakeTime()
    router = _router({"p1": FakeLLM("p1"), "p2": FakeLLM("p2")}, t)
    assert router.breaker("primary") is router.breaker("primary-alias")
    assert router.breaker("primary") is not router.breaker("backup")
    assert router.breaker("primary").name == "p1:big"
    assert router.breaker("backup").name == "p2:small"

    for _ in range(5):
        router.breaker("backup").record_failure()
    assert router.breaker("backup").state is BreakerState.OPEN
    assert router.breaker("primary").state is BreakerState.CLOSED


async def test_alias_route_sees_the_shared_open_breaker():
    t = FakeTime()
    p1 = FakeLLM("p1", default="x [1]")
    router = _router({"p1": p1, "p2": FakeLLM("p2")}, t)
    for _ in range(5):
        router.breaker("primary").record_failure()
    with pytest.raises(AllModelsFailedError):
        await router.chat("rewrite", MSGS)  # rewrite -> primary-alias -> same p1:big breaker
    assert p1.calls == []


def test_retry_policy_from_config():
    assert RetryPolicy.from_config({}) == RetryPolicy()
    p = RetryPolicy.from_config({"retry": {"max_attempts": 0, "request_deadline_s": 2}})
    assert p.max_attempts == 1 and p.request_deadline_s == 2.0
    with pytest.raises(ValueError):
        RetryPolicy.from_config({"retry": {"jitter": "equal"}})


async def test_usage_and_response_pass_through():
    class Fixed(FakeLLM):
        async def chat(self, messages, *, model, **kw) -> LLMResponse:
            return LLMResponse(text="t", model=model, provider="p1", usage=Usage(prompt_tokens=3))

    router = _router({"p1": Fixed("p1"), "p2": FakeLLM("p2")}, FakeTime())
    result = await router.chat("answer", MSGS)
    assert result.response.usage.prompt_tokens == 3 and result.attempts == 1
