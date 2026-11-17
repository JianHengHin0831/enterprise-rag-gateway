"""Local token estimation.

tiktoken is exact only for OpenAI tokenizers; for Claude / Qwen it is an estimate (non-English
text can differ noticeably). That is why the request summary records both `estimated_tokens` and
the provider-reported `actual_tokens`, plus their ratio.
"""

from functools import lru_cache

from app.gateway.base import ChatMessage

_PER_MESSAGE_OVERHEAD = 4


@lru_cache(maxsize=1)
def _encoding():  # type: ignore[no-untyped-def]
    try:
        import tiktoken

        return tiktoken.get_encoding("o200k_base")
    except Exception:  # noqa: BLE001 — offline / encoding download failure
        return None


def estimate_tokens(text: str) -> int:
    enc = _encoding()
    if enc is not None:
        return len(enc.encode(text))
    # Fallback heuristic for English text: ~4 characters per token.
    return max(1, len(text) // 4) if text else 0


def estimate_messages_tokens(messages: list[ChatMessage]) -> int:
    return sum(estimate_tokens(m.content) + _PER_MESSAGE_OVERHEAD for m in messages)
