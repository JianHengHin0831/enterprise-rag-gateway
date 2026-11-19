"""Optional LangFuse export over OpenTelemetry (OTLP/HTTP, JSON encoding).

Enabled only when RAG_LANGFUSE_PUBLIC_KEY and RAG_LANGFUSE_SECRET_KEY are both set. LangFuse's
legacy batch-ingestion API was sunset on Langfuse Cloud on 2026-11-16; the supported path is
`POST {host}/api/public/otel/v1/traces` with Basic auth and `x-langfuse-ingestion-version: 4`.

Per request: one root span `rag_request` (trace-level attributes: user, tenant, role,
decision, cost, ...) plus one child span per pipeline `Span` (rewrite / cache_lookup /
retrieve / rerank / budget / llm); the `llm` span is typed as a generation with model and
token usage. Only the already-sanitized summary and span attributes are sent: no query text,
no answer text, no document text (the audit log's query_hash rule applies here too).

Export never blocks or fails a request: `submit` schedules a background task; errors are
logged with the status code only.
"""

import asyncio
import base64
import json
import logging
import secrets
import time
from typing import Any

import httpx

from app.config import Settings
from app.observability.logging import log_event
from app.observability.sanitizer import sanitize
from app.observability.trace import SpanRecord, current_trace_id

logger = logging.getLogger("rag.langfuse")

OTLP_PATH = "/api/public/otel/v1/traces"
SERVICE_NAME = "enterprise-rag-gateway"
_OK, _ERROR = 1, 2  # OTLP StatusCode


def _attr_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}  # OTLP JSON encodes int64 as a string
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    return {"stringValue": json.dumps(value, ensure_ascii=False, default=str)}


def _attrs(items: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"key": k, "value": _attr_value(v)} for k, v in items.items() if v is not None]


def _ns(seconds: float) -> str:
    return str(int(seconds * 1_000_000_000))


def build_otlp_payload(
    trace_id: str, summary: dict[str, Any], spans: list[SpanRecord]
) -> dict[str, Any]:
    """OTLP ExportTraceServiceRequest (JSON mapping) for one finished request."""
    summary = sanitize(summary)
    root_id = secrets.token_hex(8)
    start = min((s.start_ts for s in spans), default=None)
    if start is None:
        start = time.time() - summary.get("total_ms", 0) / 1000
    end = start + (summary.get("total_ms") or 0) / 1000

    trace_meta = {
        k: v
        for k, v in summary.items()
        if k not in {"user_id", "steps_ms", "model_used"} and v is not None
    }
    root_attrs = {
        "langfuse.trace.name": "rag_request",
        "langfuse.user.id": summary.get("user_id"),
        "langfuse.trace.tags": [summary.get("tenant_id"), summary.get("decision")],
        **{f"langfuse.trace.metadata.{k}": v for k, v in trace_meta.items()},
    }
    otlp_spans = [
        {
            "traceId": trace_id,
            "spanId": root_id,
            "name": "rag_request",
            "kind": 1,
            "startTimeUnixNano": _ns(start),
            "endTimeUnixNano": _ns(end),
            "attributes": _attrs(root_attrs),
            "status": {"code": _ERROR if summary.get("error_type") else _OK},
        }
    ]
    for record in spans:
        attrs = sanitize(record.attrs)
        span_attrs: dict[str, Any] = {
            f"langfuse.observation.metadata.{k}": v for k, v in attrs.items()
        }
        if record.name == "llm":
            span_attrs.update(
                {
                    "langfuse.observation.type": "generation",
                    "gen_ai.request.model": attrs.get("model_used"),
                    "gen_ai.response.model": attrs.get("model_used"),
                    "gen_ai.usage.input_tokens": attrs.get("prompt_tokens"),
                    "gen_ai.usage.output_tokens": attrs.get("completion_tokens"),
                }
            )
        if record.status != "ok":
            span_attrs["langfuse.observation.level"] = "ERROR"
        otlp_spans.append(
            {
                "traceId": trace_id,
                "spanId": secrets.token_hex(8),
                "parentSpanId": root_id,
                "name": record.name,
                "kind": 1,
                "startTimeUnixNano": _ns(record.start_ts),
                "endTimeUnixNano": _ns(record.start_ts + record.latency_ms / 1000),
                "attributes": _attrs(span_attrs),
                "status": {"code": _OK if record.status == "ok" else _ERROR},
            }
        )
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": _attrs({"service.name": SERVICE_NAME})},
                "scopeSpans": [{"scope": {"name": "rag.pipeline"}, "spans": otlp_spans}],
            }
        ]
    }


class LangfuseExporter:
    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        host: str,
        public_key: str,
        secret_key: str,
        timeout_s: float = 5.0,
    ) -> None:
        self._http = http
        self._url = host.rstrip("/") + OTLP_PATH
        token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        self._headers = {
            "Authorization": f"Basic {token}",
            "Content-Type": "application/json",
            "x-langfuse-ingestion-version": "4",
        }
        self._timeout_s = timeout_s
        self._pending: set[asyncio.Task[None]] = set()

    def submit(self, summary: dict[str, Any], spans: list[SpanRecord]) -> None:
        trace_id = current_trace_id()
        if not trace_id or len(trace_id) != 32:
            return
        payload = build_otlp_payload(trace_id, summary, spans)
        task = asyncio.create_task(self._send(payload))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _send(self, payload: dict[str, Any]) -> None:
        try:
            r = await self._http.post(
                self._url, json=payload, headers=self._headers, timeout=self._timeout_s
            )
            if r.status_code >= 300:
                log_event(
                    logger, "langfuse_export_failed", level=logging.WARNING, status=r.status_code
                )
        except httpx.HTTPError as exc:
            log_event(
                logger, "langfuse_export_failed", level=logging.WARNING, error=type(exc).__name__
            )

    async def aclose(self, timeout_s: float = 5.0) -> None:
        """Flush in-flight exports on shutdown (bounded)."""
        if self._pending:
            await asyncio.wait(self._pending, timeout=timeout_s)


def build_exporter(settings: Settings, http: httpx.AsyncClient) -> LangfuseExporter | None:
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        return None
    return LangfuseExporter(
        http,
        host=settings.langfuse_host,
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
    )
