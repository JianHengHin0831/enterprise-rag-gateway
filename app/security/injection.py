"""Indirect prompt-injection defenses for retrieved text.

1. Retrieved chunks are wrapped in tags whose name carries a per-request random nonce, so a
   document cannot guess the closing tag.
2. Anything in the chunk that looks like one of our tags (or a generic `retrieved_doc` tag) is
   neutralized before wrapping, so `</retrieved_doc>` inside a document cannot break out.
3. `screen_text` flags common injection phrasing; used at ingest time (mark chunks
   `suspicious`) and on user queries (risk signal, not a hard block).
"""

import re
import secrets
from dataclasses import dataclass

from app.schemas import RetrievedChunk

_TAG_LIKE = re.compile(r"<\s*/?\s*(?:retrieved_doc|doc_[0-9a-f]+)\b[^>]*>", re.IGNORECASE)

_SUSPICIOUS_PATTERNS: dict[str, re.Pattern[str]] = {
    "ignore_instructions": re.compile(
        r"(ignore|disregard|forget)\s+(all\s+)?(the\s+)?(previous|above|prior|earlier)\s+"
        r"(instructions|rules|prompts?)",
        re.I,
    ),
    "role_override": re.compile(
        r"you\s+are\s+now\b|act\s+as\s+(a|an)\b|from\s+now\s+on,?\s+you\b", re.I
    ),
    # Requires an exfiltration verb near the secret; a bare mention ("key rotation") is benign.
    "secret_exfil": re.compile(
        r"\b(reveal|print|output|show|leak|send|dump)\b.{0,30}"
        r"(api[\s_-]?keys?|secret\s+keys?|system\s*prompt|passwords?)",
        re.I,
    ),
    "fake_tag": _TAG_LIKE,
}


@dataclass(frozen=True)
class WrappedDocuments:
    tag: str
    text: str


def neutralize(text: str) -> str:
    return _TAG_LIKE.sub("[removed-tag]", text)


def wrap_documents(chunks: list[RetrievedChunk]) -> WrappedDocuments:
    """Return the documents block and the tag name to reference in the system prompt.

    Document ids are 1-based and correspond to the [n] citations the model must emit.
    """
    tag = f"doc_{secrets.token_hex(4)}"
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        title = neutralize(chunk.title or chunk.doc_id).replace('"', "'")
        parts.append(f'<{tag} id="{i}" title="{title}">\n{neutralize(chunk.text)}\n</{tag}>')
    return WrappedDocuments(tag=tag, text="\n\n".join(parts))


def screen_text(text: str) -> list[str]:
    """Return names of matched injection patterns (empty list = nothing suspicious)."""
    return [name for name, pattern in _SUSPICIOUS_PATTERNS.items() if pattern.search(text)]
