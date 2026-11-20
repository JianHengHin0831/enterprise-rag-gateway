"""Optional LangFuse export over OpenTelemetry (OTLP/HTTP, JSON encoding).

Enabled only when RAG_LANGFUSE_PUBLIC_KEY and RAG_LANGFUSE_SECRET_KEY are both set. LangFuse's
legacy batch-ingestion API was sunset on Langfuse Cloud on 2026-11-16; the supported path is
`POST {host}/api/public/otel/v1/traces` with Basic auth and `x-langfuse-ingestion-version: 4`.

Per request: one root span `rag_request` plus one child observation per pipeline `Span`,
following https://langfuse.com/docs/observability/best-practices:
- each step gets its most specific observation type (`OBSERVATION_TYPES`: LLM calls are
  generations with model, usage and cost; vector search is a retriever; the rerank score gate
  is a guardrail) and a stable verb-first display name (`OBSERVATION_NAMES`);
- trace-level attributes (user, tags, environment, version) are repeated on every span so
  Langfuse can filter and aggregate observations by them;
- every observation's input/output is a structured, text-free view so a reviewer can read a
  trace at a glance: the root shows query_hash, turns, decision and chunk-level citations
  (`doc_id#chunk_index`); each step shows its key attributes (`STEP_IO`), e.g. the rerank
  gate's passed / top_score / threshold / context_refs;
- non-ok steps are marked with a level and a status message.
Only the already-sanitized summary and span attributes are sent: no query text, no answer
text, no document text (the audit log's query_hash rule applies here too).

Export never blocks or fails a request: `submit` schedules a background task; errors are
logged with the status code only.
"""

import asyncio
import base64
import json
import logging
import re
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

# Pipeline span name -> Langfuse observation type (unlisted names stay generic spans).
OBSERVATION_TYPES = {
    "rewrite": "generation",
    "llm": "generation",
    "retrieve": "retriever",
    "rerank": "guardrail",
}
# Pipeline span name -> stable, verb-first observation name shown in Langfuse.
OBSERVATION_NAMES = {
    "rewrite": "rewrite-query",
    "cache_lookup": "lookup-answer-cache",
    "retrieve": "retrieve-context",
    "rerank": "rerank-and-gate",
    "budget": "reserve-token-budget",
    "llm": "generate-answer",
}
# Pipeline span name -> (input keys, output keys): span attrs shown as the observation's
# Input / Output so each step reads at a glance (all attrs also stay in metadata for filters).
STEP_IO: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "rewrite": (("turns",), ("rewritten", "fallback", "error_type")),
    "cache_lookup": ((), ("hit",)),
    "retrieve": (("top_k",), ("hits", "retrieved_doc_ids", "suspicious_hits")),
    "rerank": (("reranker", "candidates"), ("passed", "top_score", "threshold", "context_refs")),
    "budget": ((), ("estimated_tokens", "accepted", "downgraded")),
    "llm": (("task", "contexts"), ("degraded", "attempts", "prompt_tokens", "completion_tokens")),
}
# Summary keys shown as the root observation's input / output (never raw text).
_ROOT_INPUT_KEYS = ("query_hash", "turns", "stream")
_ROOT_OUTPUT_KEYS = (
    "decision",
    "refused",
    "refusal_reason",
    "cache_hit",
    "citations",
    "citation_mismatch",
    "truncated",
    "degraded",
    "error_type",
)
_ENV_RE = re.compile(r"^(?!langfuse)[a-z0-9_-]{1,40}$")


def langfuse_environment(env: str) -> str | None:
    """Langfuse environment names must match ^(?!langfuse)[a-z0-9-_]+$ (max 40 chars)."""
    value = env.strip().lower()
    return value if _ENV_RE.match(value) else None


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
    trace_id: str,
    summary: dict[str, Any],
    spans: list[SpanRecord],
    *,
    environment: str | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    """OTLP ExportTraceServiceRequest (JSON mapping) for one finished request."""
    summary = sanitize(summary)
    root_id = secrets.token_hex(8)
    start = min((s.start_ts for s in spans), default=None)
    if start is None:
        start = time.time() - summary.get("total_ms", 0) / 1000
    end = start + (summary.get("total_ms") or 0) / 1000

    tags = [summary.get("tenant_id"), summary.get("role"), summary.get("decision")]
    # Trace-level attributes go on every span: Langfuse filters observations by them.
    trace_attrs: dict[str, Any] = {
        "langfuse.trace.name": "rag_request",
        "langfuse.user.id": summary.get("user_id"),
        "langfuse.trace.tags": [t for t in tags if t],
        "langfuse.environment": environment,
        "langfuse.version": version,
    }
    trace_meta = {
        k: v
        for k, v in summary.items()
        if k not in {"user_id", "steps_ms", "model_used"} and v is not None
    }
    error_type = summary.get("error_type")
    root_attrs: dict[str, Any] = {
        **trace_attrs,
        **{f"langfuse.trace.metadata.{k}": v for k, v in trace_meta.items()},
        "langfuse.observation.input": {k: summary.get(k) for k in _ROOT_INPUT_KEYS},
        "langfuse.observation.output": {
            **{k: summary.get(k) for k in _ROOT_OUTPUT_KEYS},
            # Chunk-level refs (`doc_id#chunk_index`) when known; doc ids otherwise.
            "citations": summary.get("cited_refs", summary.get("citations")),
        },
    }
    if error_type:
        root_attrs["langfuse.observation.level"] = "ERROR"
        root_attrs["langfuse.observation.status_message"] = error_type
    elif summary.get("refused") or summary.get("citation_mismatch") or summary.get("truncated"):
        root_attrs["langfuse.observation.level"] = "WARNING"
    otlp_spans = [
        {
            "traceId": trace_id,
            "spanId": root_id,
            "name": "rag_request",
            "kind": 1,
            "startTimeUnixNano": _ns(start),
            "endTimeUnixNano": _ns(end),
            "attributes": _attrs(root_attrs),
            "status": {"code": _ERROR if error_type else _OK},
        }
    ]
    for record in spans:
        attrs = sanitize(record.attrs)
        obs_type = OBSERVATION_TYPES.get(record.name, "span")
        span_attrs: dict[str, Any] = {
            **trace_attrs,
            "langfuse.observation.type": obs_type,
            **{f"langfuse.observation.metadata.{k}": v for k, v in attrs.items()},
        }
        in_keys, out_keys = STEP_IO.get(record.name, ((), ()))
        for attr_key, keys in (
            ("langfuse.observation.input", in_keys),
            ("langfuse.observation.output", out_keys),
        ):
            view = {k: attrs[k] for k in keys if attrs.get(k) is not None}
            if view:
                span_attrs[attr_key] = view
        if obs_type == "generation":
            span_attrs.update(
                {
                    "langfuse.observation.model.name": attrs.get("model_used"),
                    "gen_ai.request.model": attrs.get("model_used"),
                    "gen_ai.usage.input_tokens": attrs.get("prompt_tokens"),
                    "gen_ai.usage.output_tokens": attrs.get("completion_tokens"),
                }
            )
            # Model keys are gateway aliases Langfuse cannot price; send the computed cost.
            cost = attrs.get("cost_usd")
            if cost is None and record.name == "llm":
                cost = summary.get("cost_usd")
            if cost is not None:
                span_attrs["langfuse.observation.cost_details"] = {"total": cost}
        if record.status != "ok":
            span_attrs["langfuse.observation.level"] = "ERROR"
            span_attrs["langfuse.observation.status_message"] = attrs.get(
                "error_type", record.status
            )
        elif attrs.get("fallback") or attrs.get("passed") is False:
            span_attrs["langfuse.observation.level"] = "WARNING"
        otlp_spans.append(
            {
                "traceId": trace_id,
                "spanId": secrets.token_hex(8),
                "parentSpanId": root_id,
                "name": OBSERVATION_NAMES.get(record.name, record.name),
                "kind": 1,
                "startTimeUnixNano": _ns(record.start_ts),
                "endTimeUnixNano": _ns(record.start_ts + record.latency_ms / 1000),
                "attributes": _attrs(span_attrs),
                "status": {"code": _OK if record.status == "ok" else _ERROR},
            }
        )
    resource = {"service.name": SERVICE_NAME, "langfuse.environment": environment}
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": _attrs(resource)},
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
        environment: str | None = None,
        version: str | None = None,
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
        self._environment = environment
        self._version = version
        self._timeout_s = timeout_s
        self._pending: set[asyncio.Task[None]] = set()

    def submit(self, summary: dict[str, Any], spans: list[SpanRecord]) -> None:
        trace_id = current_trace_id()
        if not trace_id or len(trace_id) != 32:
            return
        payload = build_otlp_payload(
            trace_id, summary, spans, environment=self._environment, version=self._version
        )
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
        environment=langfuse_environment(settings.env),
        # Prompt versions change answers; compare traces across them in Langfuse.
        version=(
            f"answer-v{settings.answer_prompt_version}.rewrite-v{settings.rewrite_prompt_version}"
        ),
    )
