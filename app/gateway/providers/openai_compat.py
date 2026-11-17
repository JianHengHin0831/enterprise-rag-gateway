"""OpenAI-compatible Chat Completions adapter (DashScope, SiliconFlow, OpenAI, vLLM...).

TODO(M1): implement non-streaming `chat` with the shared httpx.AsyncClient:
  POST {base_url}/chat/completions, map `usage.prompt_tokens/completion_tokens`,
  map HTTP errors with `error_from_status` (parse `Retry-After`), map httpx timeouts to
  RetryableLLMError.
TODO(M2): implement `stream` (SSE `data:` lines, `[DONE]`, `stream_options.include_usage`).
"""

from collections.abc import AsyncIterator

import httpx

from app.gateway.base import BaseLLMProvider, ChatMessage, LLMResponse, StreamChunk


class OpenAICompatProvider(BaseLLMProvider):
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
        raise NotImplementedError("OpenAICompatProvider.chat — milestone M1")

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
    ) -> AsyncIterator[StreamChunk]:
        raise NotImplementedError("OpenAICompatProvider.stream — milestone M2")
