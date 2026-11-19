"""OpenAI-compatible Chat Completions adapter (DashScope, SiliconFlow, OpenAI, vLLM...).

Non-streaming `chat` uses the shared httpx.AsyncClient:
  POST {base_url}/chat/completions -> choices[0].message.content + usage.
Error mapping: HTTP status via `error_from_status` (with `Retry-After`), timeouts and
transport failures -> RetryableLLMError, malformed bodies -> NonRetryableLLMError.

TODO(M2): implement `stream` (SSE `data:` lines, `[DONE]`, `stream_options.include_usage`).
"""

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

from app.gateway.base import (
    BaseLLMProvider,
    ChatMessage,
    LLMResponse,
    NonRetryableLLMError,
    RetryableLLMError,
    StreamChunk,
    Usage,
    error_from_status,
)

_MAX_ERROR_BODY = 300


def parse_retry_after(value: str | None) -> float | None:
    """`Retry-After` is either delta-seconds or an HTTP date."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


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
        try:
            r = await self._client.post(
                f"{self._base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise RetryableLLMError(f"{self.name}: timeout after {timeout_s}s") from exc
        except httpx.TransportError as exc:
            raise RetryableLLMError(f"{self.name}: {type(exc).__name__}") from exc

        if r.status_code >= 400:
            # The body may echo our prompt; keep only a short prefix for diagnostics.
            raise error_from_status(
                r.status_code,
                f"{self.name}: HTTP {r.status_code}: {r.text[:_MAX_ERROR_BODY]}",
                parse_retry_after(r.headers.get("Retry-After")),
            )

        try:
            data = r.json()
            choice = data["choices"][0]
            text = choice["message"]["content"] or ""
            usage = data.get("usage") or {}
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise NonRetryableLLMError(f"{self.name}: malformed response") from exc

        return LLMResponse(
            text=text,
            model=data.get("model") or model,
            provider=self.name,
            usage=Usage(
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
            ),
            finish_reason=choice.get("finish_reason"),
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
        raise NotImplementedError("OpenAICompatProvider.stream — milestone M2")
