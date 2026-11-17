"""Anthropic Messages API adapter.

Dialect differences handled here (TODO(M2)):
- `system` is a top-level parameter, not a message: pull system messages out of the list.
- `max_tokens` is required.
- usage fields are `input_tokens` / `output_tokens` -> map to Usage(prompt/completion).
- Streaming events: message_start / content_block_delta / message_delta (final usage).
- 529 (overloaded) is retryable, as are 429 and 5xx.
"""

from collections.abc import AsyncIterator

import httpx

from app.gateway.base import BaseLLMProvider, ChatMessage, LLMResponse, StreamChunk


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
        temperature: float = 0.0,
        timeout_s: float = 30.0,
    ) -> LLMResponse:
        raise NotImplementedError("AnthropicProvider.chat — milestone M2")

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
