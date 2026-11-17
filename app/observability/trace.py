"""Request tracing via contextvars.

trace_id = start_trace()
async with Span("retrieve") as span:
    ...
    span.set("hits", 20)
spans = current_spans()   # for the request summary event
"""

import logging
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Self

from app.observability.logging import log_event

_logger = logging.getLogger("rag.trace")

_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_spans: ContextVar[list["SpanRecord"] | None] = ContextVar("spans", default=None)


@dataclass
class SpanRecord:
    name: str
    start_ts: float
    latency_ms: float = 0.0
    status: str = "ok"
    attrs: dict[str, Any] = field(default_factory=dict)


def start_trace(trace_id: str | None = None) -> str:
    tid = trace_id or uuid.uuid4().hex
    _trace_id.set(tid)
    _spans.set([])
    return tid


def current_trace_id() -> str | None:
    return _trace_id.get()


def current_spans() -> list[SpanRecord]:
    return list(_spans.get() or [])


def steps_ms() -> dict[str, float]:
    return {s.name: s.latency_ms for s in current_spans()}


class Span:
    def __init__(self, name: str, **attrs: Any) -> None:
        self.record = SpanRecord(name=name, start_ts=time.time(), attrs=dict(attrs))
        self._t0 = 0.0

    def set(self, key: str, value: Any) -> None:
        self.record.attrs[key] = value

    async def __aenter__(self) -> Self:
        self._t0 = time.perf_counter()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.record.latency_ms = round((time.perf_counter() - self._t0) * 1000, 2)
        if exc is not None:
            self.record.status = "error"
            self.record.attrs["error_type"] = type(exc).__name__
        spans = _spans.get()
        if spans is not None:
            spans.append(self.record)
        log_event(
            _logger,
            "span",
            span=self.record.name,
            latency_ms=self.record.latency_ms,
            status=self.record.status,
            **self.record.attrs,
        )
