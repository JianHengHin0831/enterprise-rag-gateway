"""Append-only audit log (JSONL). Separate from runtime logs.

One line per `/v1/chat` request, whatever the outcome: who / tenant / role / query_hash /
retrieved_doc_ids / decision / ts (+ trace_id, cited docs, model). Stores a hash of the
query, never the raw query text.

Append-only: the file is only ever opened in "a" mode and there is no API that rewrites or
truncates it. Writes run in a worker thread (off the event loop) behind a lock, so lines
from concurrent requests never interleave.
"""

import asyncio
import hashlib
import json
import threading
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from app.schemas import Identity


class Decision(StrEnum):
    ANSWERED = "answered"
    REFUSED_LOW_SCORE = "refused_low_score"
    REFUSED_NO_RESULTS = "refused_no_results"
    REFUSED_BY_MODEL = "refused_by_model"
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
    cited_doc_ids: list[str] = Field(default_factory=list)
    decision: Decision
    model_used: str | None = None
    degraded: bool = False
    error_type: str | None = None


def hash_query(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:32]


def build_record(
    *,
    trace_id: str,
    identity: Identity,
    query: str,
    retrieved_doc_ids: list[str],
    decision: Decision,
    cited_doc_ids: list[str] | None = None,
    model_used: str | None = None,
    degraded: bool = False,
    error_type: str | None = None,
) -> AuditRecord:
    return AuditRecord(
        trace_id=trace_id,
        user_id=identity.user_id,
        tenant_id=identity.tenant_id,
        role=identity.role.value,
        query_hash=hash_query(query),
        retrieved_doc_ids=retrieved_doc_ids,
        cited_doc_ids=cited_doc_ids or [],
        decision=decision,
        model_used=model_used,
        degraded=degraded,
        error_type=error_type,
    )


class AuditLogger:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def write(self, record: AuditRecord) -> None:
        line = json.dumps(record.model_dump(mode="json"), ensure_ascii=False)
        with self._lock, self._path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")

    async def awrite(self, record: AuditRecord) -> None:
        await asyncio.to_thread(self.write, record)
