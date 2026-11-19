"""HTTP plumbing shared by the remote provider adapters.

Maps transport failures and HTTP statuses onto the error taxonomy in `app/gateway/base.py`:
timeouts -> ProviderTimeoutError, other transport errors -> RetryableLLMError, HTTP >= 400 via
`error_from_status` (with `Retry-After`), a non-JSON body -> NonRetryableLLMError.
"""

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
