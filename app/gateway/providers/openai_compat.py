"""OpenAI-compatible Chat Completions adapter (DashScope, SiliconFlow, OpenAI, vLLM...).

Non-streaming `chat` uses the shared httpx.AsyncClient:
  POST {base_url}/chat/completions -> choices[0].message.content + usage.
Streaming `stream` sends `stream: true` + `stream_options.include_usage` and reads SSE
`data:` lines until `[DONE]`: text in `choices[0].delta.content`, usage in the last event
(which has empty `choices`).
Error mapping: see `app/gateway/providers/_http.py`; `finish_reason == "content_filter"`
-> ContentRefusalError, malformed bodies -> NonRetryableLLMError.
"""

from collections.abc import AsyncIterator

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
from app.gateway.providers._http import open_sse, parse_retry_after, post_json

__all__ = ["OpenAICompatProvider", "parse_retry_after"]


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
        body = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        data = await post_json(
            self._client,
            f"{self._base_url}/chat/completions",
            provider=self.name,
            body=body,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout_s=timeout_s,
        )
        try:
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
            usage = data.get("usage") or {}
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise NonRetryableLLMError(f"{self.name}: malformed response") from exc

        if finish_reason == "content_filter":
            raise ContentRefusalError(f"{self.name}: response blocked by content filter")

        return LLMResponse(
            text=text,
            model=data.get("model") or model,
            provider=self.name,
            usage=Usage(
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
            ),
            finish_reason=finish_reason,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
    ) -> AsyncIterator[StreamChunk]:
        body = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        usage: Usage | None = None
        finish_reason: str | None = None
        async with open_sse(
            self._client,
            f"{self._base_url}/chat/completions",
            provider=self.name,
            body=body,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout_s=timeout_s,
        ) as events:
            async for event in events:
                delta = ""
                try:
                    if event.get("usage"):
                        u = event["usage"]
                        usage = Usage(
                            prompt_tokens=int(u.get("prompt_tokens") or 0),
                            completion_tokens=int(u.get("completion_tokens") or 0),
                        )
                    choices = event.get("choices") or []
                    if choices:
                        choice = choices[0]
                        delta = (choice.get("delta") or {}).get("content") or ""
                        finish_reason = choice.get("finish_reason") or finish_reason
                except (KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
                    raise NonRetryableLLMError(f"{self.name}: malformed stream event") from exc
                if finish_reason == "content_filter":
                    raise ContentRefusalError(f"{self.name}: response blocked by content filter")
                if delta:
                    yield StreamChunk(delta=delta)
        yield StreamChunk(usage=usage or Usage(), finish_reason=finish_reason)
