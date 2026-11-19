"""Anthropic Messages API adapter (raw HTTP on the shared httpx.AsyncClient).

Raw HTTP instead of the `anthropic` SDK on purpose: the router owns retries, backoff and
breakers, and all providers share one connection pool created in the FastAPI lifespan.

Dialect differences handled here:
- `system` is a top-level parameter, not a message: system messages are pulled out of the
  list and joined.
- `max_tokens` is required.
- `temperature` is NOT sent: current Claude models reject non-default sampling parameters.
- The response is a list of content blocks; only `text` blocks form the answer (`thinking`
  blocks are skipped).
- usage fields are `input_tokens` (+ cache creation / read) / `output_tokens`
  -> Usage(prompt_tokens / completion_tokens).
- `stop_reason == "refusal"` (HTTP 200) -> ContentRefusalError.
- 529 (overloaded) is retryable, as are 429 and 5xx (see `error_from_status`).

TODO(M2): `stream` (events message_start / content_block_delta / message_delta).
"""

from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.gateway.base import (
    BaseLLMProvider,
    ChatMessage,
    ContentRefusalError,
    LLMResponse,
    NonRetryableLLMError,
    StreamChunk,
    Usage,
)
from app.gateway.providers._http import post_json

API_VERSION = "2023-06-01"


def split_system(messages: list[ChatMessage]) -> tuple[str, list[dict[str, str]]]:
    """Return (system text, user/assistant turns) in the Messages API shape."""
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    turns = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
    return system, turns


class AnthropicProvider(BaseLLMProvider):
    def __init__(self, *, name: str, base_url: str, api_key: str, client: httpx.AsyncClient):
        self.name = name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = client

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,  # accepted for interface parity, never sent
        timeout_s: float = 30.0,
    ) -> LLMResponse:
        system, turns = split_system(messages)
        if not turns or turns[0]["role"] != "user":
            raise NonRetryableLLMError(f"{self.name}: conversation must start with a user turn")
        body: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": turns}
        if system:
            body["system"] = system

        data = await post_json(
            self._client,
            f"{self._base_url}/v1/messages",
            provider=self.name,
            body=body,
            headers={"x-api-key": self._api_key, "anthropic-version": API_VERSION},
            timeout_s=timeout_s,
        )
        try:
            stop_reason = data.get("stop_reason")
            text = "".join(
                block["text"] for block in data["content"] if block.get("type") == "text"
            )
            usage = data.get("usage") or {}
            prompt_tokens = sum(
                int(usage.get(field) or 0)
                for field in (
                    "input_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                )
            )
            completion_tokens = int(usage.get("output_tokens") or 0)
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            raise NonRetryableLLMError(f"{self.name}: malformed response") from exc

        if stop_reason == "refusal":
            details = data.get("stop_details") or {}
            category = details.get("category") if isinstance(details, dict) else None
            raise ContentRefusalError(f"{self.name}: model refused (category={category})")

        return LLMResponse(
            text=text,
            model=data.get("model") or model,
            provider=self.name,
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
            finish_reason=stop_reason,
        )

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
    ) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError("AnthropicProvider.stream — milestone M2")
