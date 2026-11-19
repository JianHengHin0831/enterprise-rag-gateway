"""HTTP plumbing shared by the remote provider adapters.

Maps transport failures and HTTP statuses onto the error taxonomy in `app/gateway/base.py`:
timeouts -> ProviderTimeoutError, other transport errors -> RetryableLLMError, HTTP >= 400 via
`error_from_status` (with `Retry-After`), a non-JSON body -> NonRetryableLLMError.
`open_sse` does the same for streaming (Server-Sent Events) responses.
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from app.gateway.base import (
    NonRetryableLLMError,
    ProviderTimeoutError,
    RetryableLLMError,
    error_from_status,
)

# Error bodies may echo our prompt; keep only a short prefix for diagnostics.
MAX_ERROR_BODY = 300


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


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    provider: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout_s: float,
) -> Any:
    try:
        r = await client.post(url, json=body, headers=headers, timeout=timeout_s)
    except httpx.TimeoutException as exc:
        raise ProviderTimeoutError(f"{provider}: timeout after {timeout_s}s") from exc
    except httpx.TransportError as exc:
        raise RetryableLLMError(f"{provider}: {type(exc).__name__}") from exc

    if r.status_code >= 400:
        raise error_from_status(
            r.status_code,
            f"{provider}: HTTP {r.status_code}: {r.text[:MAX_ERROR_BODY]}",
            parse_retry_after(r.headers.get("Retry-After")),
        )
    try:
        return r.json()
    except ValueError as exc:
        raise NonRetryableLLMError(f"{provider}: malformed response") from exc


@asynccontextmanager
async def open_sse(
    client: httpx.AsyncClient,
    url: str,
    *,
    provider: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout_s: float,
) -> AsyncIterator[AsyncIterator[dict[str, Any]]]:
    """POST a streaming request and yield an iterator over the JSON payloads of its SSE
    `data:` lines (`[DONE]` ends it). Opening raises the same taxonomy as `post_json`; a
    failure while reading the body is mapped too, so callers only ever see LLMError."""
    try:
        async with client.stream("POST", url, json=body, headers=headers, timeout=timeout_s) as r:
            if r.status_code >= 400:
                text = (await r.aread()).decode("utf-8", errors="replace")
                raise error_from_status(
                    r.status_code,
                    f"{provider}: HTTP {r.status_code}: {text[:MAX_ERROR_BODY]}",
                    parse_retry_after(r.headers.get("Retry-After")),
                )
            yield _sse_payloads(r, provider)
    except httpx.TimeoutException as exc:
        raise ProviderTimeoutError(f"{provider}: stream timeout after {timeout_s}s") from exc
    except httpx.TransportError as exc:
        raise RetryableLLMError(f"{provider}: {type(exc).__name__}") from exc


async def _sse_payloads(r: httpx.Response, provider: str) -> AsyncIterator[dict[str, Any]]:
    async for line in r.aiter_lines():
        if not line.startswith("data:"):
            continue  # `event:` names, comments, keep-alives, blank separators
        data = line[5:].strip()
        if not data:
            continue
        if data == "[DONE]":
            return
        try:
            payload = json.loads(data)
        except ValueError as exc:
            raise NonRetryableLLMError(f"{provider}: malformed stream event") from exc
        if isinstance(payload, dict):
            yield payload
