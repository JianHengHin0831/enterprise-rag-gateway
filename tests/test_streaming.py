"""Streaming (M2): provider SSE parsing, router pre-first-token fallback, /v1/chat SSE."""

import asyncio
import json

import httpx
import pytest

from app.gateway.base import (
    AllModelsFailedError,
    ChatMessage,
    ContentRefusalError,
    NonRetryableLLMError,
    ProviderTimeoutError,
    RateLimitedError,
    RetryableLLMError,
    StreamChunk,
    Usage,
)
from app.gateway.circuit_breaker import BreakerState
from app.gateway.providers.anthropic import AnthropicProvider
from app.gateway.providers.fake import FakeLLM
from app.gateway.providers.openai_compat import OpenAICompatProvider
from app.gateway.providers.template import FALLBACK_TEXT, TemplateProvider
from app.gateway.router import ModelRouter

MSGS = [ChatMessage(role="system", content="sys"), ChatMessage(role="user", content="hi")]


def _sse(*events: object, done: bool = False) -> bytes:
    lines = []
    for e in events:
        if isinstance(e, tuple):
            name, data = e
            lines.append(f"event: {name}\ndata: {json.dumps(data)}\n\n")
        else:
            lines.append(f"data: {json.dumps(e)}\n\n")
    if done:
        lines.append("data: [DONE]\n\n")
    return ("".join(lines)).encode()


async def _collect(stream) -> tuple[str, list[StreamChunk]]:
    chunks = [c async for c in stream]
    return "".join(c.delta for c in chunks), chunks


# ---------- OpenAI-compatible stream ----------


def _openai(handler) -> tuple[OpenAICompatProvider, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAICompatProvider(
        name="dashscope", base_url="https://llm.example/v1", api_key="k", client=client
    ), client


async def test_openai_stream_parses_deltas_and_final_usage():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        body = _sse(
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "Answer "}}]},
            {"choices": [{"delta": {"content": "[1]"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 3}},
            done=True,
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    provider, client = _openai(handler)
    async with client:
        text, chunks = await _collect(provider.stream(MSGS, model="m", max_tokens=8))
    assert seen["body"]["stream"] is True
    assert seen["body"]["stream_options"] == {"include_usage": True}
    assert text == "Answer [1]"
    assert chunks[-1].usage == Usage(prompt_tokens=50, completion_tokens=3)
    assert chunks[-1].finish_reason == "stop"


async def test_openai_stream_http_error_is_classified_before_any_chunk():
    provider, client = _openai(
        lambda _: httpx.Response(429, headers={"Retry-After": "2"}, text="slow down")
    )
    async with client:
        with pytest.raises(RateLimitedError) as info:
            await _collect(provider.stream(MSGS, model="m", max_tokens=8))
    assert info.value.retry_after_s == 2.0


async def test_openai_stream_content_filter_is_a_refusal():
    body = _sse({"choices": [{"delta": {}, "finish_reason": "content_filter"}]}, done=True)
    provider, client = _openai(lambda _: httpx.Response(200, content=body))
    async with client:
        with pytest.raises(ContentRefusalError):
            await _collect(provider.stream(MSGS, model="m", max_tokens=8))


async def test_openai_mid_stream_error_is_raised_not_a_truncated_answer():
    body = _sse(
        {"choices": [{"delta": {"content": "Partial "}}]},
        {
            "error": {"code": 502, "message": "upstream died"},
            "choices": [{"delta": {"content": ""}, "finish_reason": "error"}],
        },
        done=True,
    )
    provider, client = _openai(
        lambda _: httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})
    )
    seen: list[str] = []
    async with client:
        with pytest.raises(RetryableLLMError) as info:
            async for chunk in provider.stream(MSGS, model="m", max_tokens=8):
                seen.append(chunk.delta)
    assert seen == ["Partial "] and info.value.status_code == 502


async def test_openai_stream_malformed_event_is_non_retryable():
    provider, client = _openai(lambda _: httpx.Response(200, content=b"data: {nope\n\n"))
    async with client:
        with pytest.raises(NonRetryableLLMError, match="malformed"):
            await _collect(provider.stream(MSGS, model="m", max_tokens=8))


async def test_stream_read_timeout_is_retryable():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    provider, client = _openai(handler)
    async with client:
        with pytest.raises(ProviderTimeoutError):
            await _collect(provider.stream(MSGS, model="m", max_tokens=8))


# ---------- Anthropic stream ----------


def _anthropic(handler) -> tuple[AnthropicProvider, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return AnthropicProvider(
        name="anthropic", base_url="https://claude.example", api_key="k", client=client
    ), client


async def test_anthropic_stream_parses_text_deltas_and_usage():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_sse(
                (
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {"usage": {"input_tokens": 40, "cache_read_input_tokens": 2}},
                    },
                ),
                ("content_block_start", {"type": "content_block_start", "index": 0}),
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "delta": {"type": "thinking_delta", "thinking": "hmm"},
                    },
                ),
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": "Answer "},
                    },
                ),
                ("ping", {"type": "ping"}),
                (
                    "content_block_delta",
                    {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "[1]"}},
                ),
                (
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": 6},
                    },
                ),
                ("message_stop", {"type": "message_stop"}),
            ),
        )

    provider, client = _anthropic(handler)
    async with client:
        text, chunks = await _collect(provider.stream(MSGS, model="m", max_tokens=8))
    assert seen["body"]["stream"] is True and seen["body"]["system"] == "sys"
    assert text == "Answer [1]"  # thinking delta skipped
    assert chunks[-1].usage == Usage(prompt_tokens=42, completion_tokens=6)
    assert chunks[-1].finish_reason == "end_turn"


@pytest.mark.parametrize(
    ("err_type", "error_cls", "retryable"),
    [
        ("overloaded_error", RetryableLLMError, True),
        ("rate_limit_error", RateLimitedError, True),
        ("invalid_request_error", NonRetryableLLMError, False),
    ],
)
async def test_anthropic_in_stream_error_event_is_classified(err_type, error_cls, retryable):
    body = _sse(("error", {"type": "error", "error": {"type": err_type, "message": "x"}}))
    provider, client = _anthropic(lambda _: httpx.Response(200, content=body))
    async with client:
        with pytest.raises(error_cls) as info:
            await _collect(provider.stream(MSGS, model="m", max_tokens=8))
    assert info.value.retryable is retryable


async def test_anthropic_stream_refusal():
    body = _sse(
        ("message_start", {"type": "message_start", "message": {"usage": {}}}),
        (
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "refusal"}, "usage": {}},
        ),
    )
    provider, client = _anthropic(lambda _: httpx.Response(200, content=body))
    async with client:
        with pytest.raises(ContentRefusalError):
            await _collect(provider.stream(MSGS, model="m", max_tokens=8))


async def test_anthropic_stream_529_before_stream_is_retryable():
    provider, client = _anthropic(lambda _: httpx.Response(529, json={"type": "error"}))
    async with client:
        with pytest.raises(RetryableLLMError) as info:
            await _collect(provider.stream(MSGS, model="m", max_tokens=8))
    assert info.value.status_code == 529


# ---------- Router.open_stream ----------

CONFIG = {
    "models": {
        "primary": {"provider": "p1", "model": "big", "timeout_s": 5, "max_output_tokens": 64},
        "backup": {"provider": "p2", "model": "small", "timeout_s": 5, "max_output_tokens": 64},
        "template": {"provider": "local", "model": "tpl", "timeout_s": 1, "max_output_tokens": 0},
    },
    "routes": {"answer": ["primary", "backup", "template"], "rewrite": ["backup"]},
    "circuit_breaker": {"failure_threshold": 5, "recovery_timeout_s": 30, "half_open_max_calls": 1},
    "retry": {"max_attempts": 2, "base_delay_s": 0, "max_delay_s": 0, "request_deadline_s": 5},
}


class BreaksAfter(FakeLLM):
    """Streams `n_ok` chunks, then raises `error` (a mid-answer failure)."""

    def __init__(self, name: str, n_ok: int, error: Exception) -> None:
        super().__init__(name)
        self.n_ok, self.error = n_ok, error

    async def stream(self, messages, *, model, **kw):
        self.calls.append(model)
        for i in range(self.n_ok):
            yield StreamChunk(delta=f"part{i} ")
        raise self.error


class Stalls(FakeLLM):
    def __init__(self, name: str, first: bool) -> None:
        super().__init__(name)
        self.first = first
        self.closed = False

    async def stream(self, messages, *, model, **kw):
        self.calls.append(model)
        try:
            if not self.first:
                yield StreamChunk(delta="hello ")
            await asyncio.sleep(10)
            yield StreamChunk(delta="never")
        finally:
            self.closed = True


async def test_stream_retries_and_falls_back_before_first_token():
    p1 = FakeLLM("p1", script=[RetryableLLMError("503", status_code=503)] * 2)
    p2 = FakeLLM("p2", default="backup answer [1]")
    router = ModelRouter(config=CONFIG, providers={"p1": p1, "p2": p2})
    routed = await router.open_stream("answer", MSGS)
    assert routed.model_key == "backup" and routed.degraded and routed.attempts == 3
    text, _ = await _collect(routed.chunks)
    assert text == "backup answer [1]"
    assert len(p1.calls) == 2


async def test_stream_failure_after_first_token_is_raised_not_switched():
    p1 = BreaksAfter("p1", n_ok=2, error=RetryableLLMError("503", status_code=503))
    p2 = FakeLLM("p2")
    router = ModelRouter(config=CONFIG, providers={"p1": p1, "p2": p2})
    routed = await router.open_stream("answer", MSGS)
    assert routed.model_key == "primary" and not routed.degraded
    received = []
    with pytest.raises(RetryableLLMError):
        async for chunk in routed.chunks:
            received.append(chunk.delta)
    assert received == ["part0 ", "part1 "]
    assert p2.calls == [] and len(p1.calls) == 1  # never retried or switched mid-answer
    assert router.breaker("primary")._consecutive_failures == 1  # counted


async def test_stream_failing_before_its_first_chunk_falls_back():
    p1 = BreaksAfter("p1", n_ok=0, error=RetryableLLMError("503", status_code=503))
    router = ModelRouter(config=CONFIG, providers={"p1": p1, "p2": FakeLLM("p2", default="b [1]")})
    routed = await router.open_stream("answer", MSGS)
    assert routed.model_key == "backup"
    assert len(p1.calls) == 2  # retried: nothing had been sent yet


async def test_stream_non_retryable_before_first_token_is_raised():
    p1 = FakeLLM("p1", script=[NonRetryableLLMError("400", status_code=400)])
    p2 = FakeLLM("p2")
    router = ModelRouter(config=CONFIG, providers={"p1": p1, "p2": p2})
    with pytest.raises(NonRetryableLLMError):
        await router.open_stream("answer", MSGS)
    assert p2.calls == [] and router.breaker("primary").state is BreakerState.CLOSED


async def test_stream_no_first_token_within_timeout_falls_back():
    p1 = Stalls("p1", first=True)
    cfg = {
        **CONFIG,
        "models": {
            **CONFIG["models"],
            "primary": {**CONFIG["models"]["primary"], "timeout_s": 0.05},
        },
    }
    router = ModelRouter(config=cfg, providers={"p1": p1, "p2": FakeLLM("p2", default="b [1]")})
    routed = await router.open_stream("answer", MSGS)
    assert routed.model_key == "backup" and p1.closed


async def test_stream_stall_after_first_token_ends_with_timeout_error():
    p1 = Stalls("p1", first=False)
    cfg = {
        **CONFIG,
        "models": {
            **CONFIG["models"],
            "primary": {**CONFIG["models"]["primary"], "timeout_s": 0.05},
        },
    }
    router = ModelRouter(config=cfg, providers={"p1": p1, "p2": FakeLLM("p2")})
    routed = await router.open_stream("answer", MSGS)
    with pytest.raises(ProviderTimeoutError, match="stalled"):
        await _collect(routed.chunks)
    assert p1.closed


async def test_stream_all_models_failing_raises_before_any_chunk():
    p1 = FakeLLM("p1", script=[RetryableLLMError("503", status_code=503)] * 2)
    p2 = FakeLLM("p2", script=[RetryableLLMError("503", status_code=503)] * 2)
    router = ModelRouter(
        config={**CONFIG, "routes": {"answer": ["primary", "backup"], "rewrite": ["backup"]}},
        providers={"p1": p1, "p2": p2},
    )
    with pytest.raises(AllModelsFailedError):
        await router.open_stream("answer", MSGS)


async def test_template_stream_is_degraded():
    router = ModelRouter(config=CONFIG, providers={"local": TemplateProvider()})
    routed = await router.open_stream("answer", MSGS)
    text, _ = await _collect(routed.chunks)
    assert routed.degraded and text == FALLBACK_TEXT


# ---------- /v1/chat with stream=true ----------


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        name = data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        events.append((name, data))
    return events
