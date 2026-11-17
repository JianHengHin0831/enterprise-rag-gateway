"""PII and secret masking for anything that goes to logs.

The JSON log formatter runs every field through `sanitize`, so callers normally don't need to
call it directly.
"""

import re
from typing import Any

_REDACTED = "[REDACTED]"

_SENSITIVE_KEYS = frozenset(
    {"authorization", "api_key", "apikey", "password", "secret", "token", "jwt", "cookie"}
)

# Order matters: 18-digit ID numbers before 11-digit phone numbers.
_RULES: list[tuple[re.Pattern[str], str]] = [
    # Mainland China resident ID: keep first 6 and last 4.
    (re.compile(r"(?<!\d)(\d{6})\d{8}(\d{3}[\dXx])(?!\d)"), r"\1********\2"),
    # Mainland China mobile: 138****1234
    (re.compile(r"(?<!\d)(1[3-9]\d)\d{4}(\d{4})(?!\d)"), r"\1****\2"),
    # Email: keep first char + domain.
    (
        re.compile(r"([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"),
        r"\1***\2",
    ),
    # API-key-looking tokens (sk-..., sk-ant-...).
    (re.compile(r"\b(sk-[A-Za-z0-9]{2})[A-Za-z0-9_-]{8,}"), r"\1****"),
    # Bearer tokens.
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"), r"\1****"),
]


def mask_text(text: str) -> str:
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text


def sanitize(value: Any, _key: str | None = None) -> Any:
    if _key is not None and _key.lower() in _SENSITIVE_KEYS:
        return _REDACTED
    if isinstance(value, str):
        return mask_text(value)
    if isinstance(value, dict):
        return {k: sanitize(v, str(k)) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [sanitize(v) for v in value]
    return value
