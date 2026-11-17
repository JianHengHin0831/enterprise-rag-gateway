"""Append-only audit log (JSONL). Separate from runtime logs.

Stores a hash of the query, never the raw query text.
"""

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from app.schemas import Identity


class Decision(StrEnum):
    ANSWERED = "answered"
    REFUSED_LOW_SCORE = "refused_low_score"
    REFUSED_NO_RESULTS = "refused_no_results"
    CACHE_HIT = "cache_hit"
    BUDGET_EXCEEDED = "budget_exceeded"
    ERROR = "error"


class AuditRecord(BaseModel):
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    trace_id: str
    user_id: str
    tenant_id: str
    role: str
    query_hash: str
    retrieved_doc_ids: list[str]
    decision: Decision
    model_used: str | None = None
    degraded: bool = False


def hash_query(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:32]


def build_record(
    *,
    trace_id: str,
    identity: Identity,
    query: str,
    retrieved_doc_ids: list[str],
    decision: Decision,
    model_used: str | None = None,
    degraded: bool = False,
) -> AuditRecord:
    return AuditRecord(
        trace_id=trace_id,
        user_id=identity.user_id,
        tenant_id=identity.tenant_id,
        role=identity.role.value,
        query_hash=hash_query(query),
        retrieved_doc_ids=retrieved_doc_ids,
        decision=decision,
        model_used=model_used,
        degraded=degraded,
    )


class AuditLogger:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: AuditRecord) -> None:
        # TODO(M3): move off the event loop (queue + background writer) if latency matters.
        line = json.dumps(record.model_dump(mode="json"), ensure_ascii=False)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
