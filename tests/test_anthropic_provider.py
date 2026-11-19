"""Anthropic Messages adapter against a mocked transport (no network)."""

import json

import httpx
import pytest

from app.gateway.base import (
    ChatMessage,
    ContentRefusalError,
    NonRetryableLLMError,
    ProviderTimeoutError,
    RateLimitedError,
    RetryableLLMError,
)
from app.gateway.providers.anthropic import API_VERSION, AnthropicProvider, split_system

MSGS = [
    ChatMessage(role="system", content="sys A"),
    ChatMessage(role="user", content="hi"),
    ChatMessage(role="assistant", content="hello"),
    ChatMessage(role="system", content="sys B"),
    ChatMessage(role="user", content="question"),
]


def _provider(handler) -> tuple[AnthropicProvider, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return (
        AnthropicProvider(
            name="anthropic", base_url="https://claude.example/", api_key="k-test", client=client
        ),
        client,
    )


def _ok(**overrides) -> dict:
    body = {
        "model": "claude-sonnet-5-5",
        "content": [
            {"type": "thinking", "thinking": ""},
            {"type": "text", "text": "Answer "},
            {"type": "text", "text": "[1]"},
        ],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 100,
            "cache_creation_input_tokens": 20,
            "cache_read_input_tokens": 5,
            "output_tokens": 9,
        },
    }
    body.update(overrides)
    return body


def test_split_system_pulls_system_out_of_turns():
    system, turns = split_system(MSGS)
    assert system == "sys A\n\nsys B"
    assert [t["role"] for t in turns] == ["user", "assistant", "user"]


async def test_chat_request_shape_and_response_mapping():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = request.headers
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_ok())

    provider, client = _provider(handler)
    async with client:
        r = await provider.chat(MSGS, model="claude-sonnet-5-5", max_tokens=64, temperature=0.3)

    assert seen["url"] == "https://claude.example/v1/messages"
    assert seen["headers"]["x-api-key"] == "k-test"
    assert seen["headers"]["anthropic-version"] == API_VERSION
    assert "authorization" not in seen["headers"]
    body = seen["body"]
    assert body["system"] == "sys A\n\nsys B"
    assert all(m["role"] != "system" for m in body["messages"])
    assert body["max_tokens"] == 64 and body["model"] == "claude-sonnet-5-5"
    assert "temperature" not in body  # current models reject non-default sampling params

    assert r.text == "Answer [1]"  # thinking block skipped, text blocks joined
    assert (r.usage.prompt_tokens, r.usage.completion_tokens) == (125, 9)
    assert r.provider == "anthropic" and r.finish_reason == "end_turn"


async def test_no_system_parameter_when_there_is_no_system_message():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_ok())

    provider, client = _provider(handler)
    async with client:
        await provider.chat([ChatMessage(role="user", content="hi")], model="m", max_tokens=8)
    assert "system" not in seen["body"]


async def test_conversation_must_start_with_user_turn():
    provider, client = _provider(lambda _: httpx.Response(200, json=_ok()))
    async with client:
        with pytest.raises(NonRetryableLLMError):
            await provider.chat([ChatMessage(role="system", content="s")], model="m", max_tokens=8)


async def test_refusal_stop_reason_is_non_retryable_content_refusal():
    payload = _ok(
        content=[], stop_reason="refusal", stop_details={"type": "refusal", "category": "cyber"}
    )
    provider, client = _provider(lambda _: httpx.Response(200, json=payload))
    async with client:
        with pytest.raises(ContentRefusalError, match="cyber") as info:
            await provider.chat(MSGS, model="m", max_tokens=8)
    assert info.value.retryable is False


@pytest.mark.parametrize(
    ("status", "error_type", "retryable"),
    [
        (429, RateLimitedError, True),
        (500, RetryableLLMError, True),
        (529, RetryableLLMError, True),  # overloaded
        (400, NonRetryableLLMError, False),
        (401, NonRetryableLLMError, False),
        (403, NonRetryableLLMError, False),
    ],
)
async def test_http_errors_are_classified(status, error_type, retryable):
    err = {"type": "error", "error": {"type": "x", "message": "y" * 1000}}
    provider, client = _provider(
        lambda _: httpx.Response(status, headers={"retry-after": "3"}, json=err)
    )
    async with client:
        with pytest.raises(error_type) as info:
            await provider.chat(MSGS, model="m", max_tokens=8)
    assert info.value.retryable is retryable
    assert info.value.status_code == status
    assert len(str(info.value)) < 400
    if status == 429:
        assert info.value.retry_after_s == 3.0


async def test_timeout_is_retryable():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    provider, client = _provider(handler)
    async with client:
        with pytest.raises(ProviderTimeoutError):
            await provider.chat(MSGS, model="m", max_tokens=8, timeout_s=1)


@pytest.mark.parametrize("body", [b"not json", b'{"content": "nope"}', b'{"foo": 1}'])
async def test_malformed_body_is_non_retryable(body):
    provider, client = _provider(lambda _: httpx.Response(200, content=body))
    async with client:
        with pytest.raises(NonRetryableLLMError, match="malformed"):
            await provider.chat(MSGS, model="m", max_tokens=8)
