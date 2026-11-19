"""OpenAI-compatible adapter against a mocked transport (no network)."""

import json
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest

from app.gateway.base import (
    ChatMessage,
    NonRetryableLLMError,
    RateLimitedError,
    RetryableLLMError,
)
from app.gateway.providers.openai_compat import OpenAICompatProvider, parse_retry_after

MSGS = [ChatMessage(role="system", content="sys"), ChatMessage(role="user", content="hi")]


def _provider(handler) -> tuple[OpenAICompatProvider, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return (
        OpenAICompatProvider(
            name="dashscope", base_url="https://llm.example/v1/", api_key="k-test", client=client
        ),
        client,
    )


async def test_chat_success_maps_text_usage_and_request():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "qwen-plus-2026",
                "choices": [{"message": {"content": "Answer [1]"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 7},
            },
        )

    provider, client = _provider(handler)
    async with client:
        r = await provider.chat(MSGS, model="qwen-plus", max_tokens=64)
    assert seen["url"] == "https://llm.example/v1/chat/completions"
    assert seen["auth"] == "Bearer k-test"
    assert seen["body"]["messages"] == [m.model_dump() for m in MSGS]
    assert seen["body"]["max_tokens"] == 64 and seen["body"]["stream"] is False
    assert r.text == "Answer [1]" and r.model == "qwen-plus-2026" and r.provider == "dashscope"
    assert (r.usage.prompt_tokens, r.usage.completion_tokens) == (120, 7)
    assert r.finish_reason == "stop"


@pytest.mark.parametrize(
    ("status", "error_type", "retryable"),
    [
        (429, RateLimitedError, True),
        (500, RetryableLLMError, True),
        (503, RetryableLLMError, True),
        (400, NonRetryableLLMError, False),
        (401, NonRetryableLLMError, False),
    ],
)
async def test_http_errors_are_classified(status, error_type, retryable):
    provider, client = _provider(
        lambda _: httpx.Response(status, headers={"Retry-After": "2"}, text="x" * 1000)
    )
    async with client:
        with pytest.raises(error_type) as info:
            await provider.chat(MSGS, model="m", max_tokens=8)
    assert info.value.retryable is retryable
    assert info.value.status_code == status
    assert len(str(info.value)) < 400  # error body is truncated
    if status == 429:
        assert info.value.retry_after_s == 2.0


async def test_timeout_is_retryable():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    provider, client = _provider(handler)
    async with client:
        with pytest.raises(RetryableLLMError, match="timeout"):
            await provider.chat(MSGS, model="m", max_tokens=8, timeout_s=1)


async def test_connection_error_is_retryable():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    provider, client = _provider(handler)
    async with client:
        with pytest.raises(RetryableLLMError):
            await provider.chat(MSGS, model="m", max_tokens=8)


@pytest.mark.parametrize("body", [b"not json", b'{"choices": []}', b'{"foo": 1}'])
async def test_malformed_body_is_non_retryable(body):
    provider, client = _provider(lambda _: httpx.Response(200, content=body))
    async with client:
        with pytest.raises(NonRetryableLLMError, match="malformed"):
            await provider.chat(MSGS, model="m", max_tokens=8)


def test_parse_retry_after():
    assert parse_retry_after(None) is None
    assert parse_retry_after("3") == 3.0
    assert parse_retry_after("-1") == 0.0
    assert parse_retry_after("garbage") is None
    future = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)
    assert 25 <= parse_retry_after(future) <= 31
