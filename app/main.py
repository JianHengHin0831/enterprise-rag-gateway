import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import anyio
import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from redis.asyncio import Redis
from redis.exceptions import RedisError
from starlette.types import Receive, Scope, Send

from app.cache.layered import LayeredCache
from app.config import Settings, get_settings
from app.cost.budget import BudgetExceededError, BudgetGuard
from app.gateway.base import AllModelsFailedError, LLMError
from app.gateway.factory import build_providers
from app.gateway.router import ModelRouter, load_models_config
from app.observability import metrics
from app.observability.langfuse import build_exporter
from app.observability.logging import configure_logging, log_event
from app.observability.trace import current_trace_id, start_trace
from app.rag.embeddings import Embedder, build_embedder
from app.rag.pipeline import RagPipeline, StreamEvent
from app.rag.prompts import load_prompt
from app.rag.reranker import Reranker, build_reranker
from app.rag.retriever import HybridRetriever
from app.rag.rewrite import QueryRewriter
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DevTokenRequest,
    DevTokenResponse,
    Identity,
    RetrievedChunk,
)
from app.security.audit import AuditLogger
from app.security.auth import get_identity, issue_token
from app.security.injection import screen_text

logger = logging.getLogger("rag.api")

WEB_DIR = Path(__file__).parent / "web"
DEV_TOKEN_TTL = timedelta(hours=1)
STREAM_CLEANUP_TIMEOUT_S = 10.0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    # Honour a test override of get_settings here too, not only in request dependencies.
    settings = app.dependency_overrides.get(get_settings, get_settings)()
    app.state.http = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=5.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
    app.state.redis = Redis.from_url(settings.redis_url)
    # No network on construction; /ready reports whether Qdrant is reachable.
    app.state.qdrant = AsyncQdrantClient(url=settings.qdrant_url, check_compatibility=False)
    embedder = build_embedder(settings)
    reranker = build_reranker(settings)
    router = ModelRouter(
        config=(models_config := load_models_config(settings.models_config_path)),
        providers=build_providers(models_config, settings, app.state.http),
    )
    budget = BudgetGuard.from_config(app.state.redis, models_config)
    exporter = build_exporter(settings, app.state.http)
    app.state.pipeline = RagPipeline(
        retriever=HybridRetriever(app.state.qdrant, embedder, settings.qdrant_collection),
        router=router,
        prompt=load_prompt(settings.prompts_dir, "rag_qa", settings.answer_prompt_version),
        retrieve_top_k=settings.retrieve_top_k,
        context_top_n=settings.rerank_top_n,
        reranker=reranker,
        rerank_threshold=settings.rerank_threshold,
        audit=AuditLogger(settings.audit_log_path),
        exporter=exporter,
        budget=budget,
        cache=LayeredCache(
            app.state.redis,
            l1_size=settings.l1_cache_size,
            l1_ttl_s=settings.l1_cache_ttl_s,
            l2_ttl_s=settings.l2_cache_ttl_s,
            index_version_ttl_s=settings.index_version_cache_ttl_s,
        ),
        rewriter=QueryRewriter(
            router=router,
            prompt=load_prompt(settings.prompts_dir, "rewrite", settings.rewrite_prompt_version),
            timeout_s=settings.rewrite_timeout_s,
            min_words=settings.rewrite_min_words,
            budget=budget,
        ),
    )
    # Load the embedding model in the background so the first request does not pay for it.
    warmup = asyncio.create_task(_warm_up(embedder, reranker))
    try:
        yield
    finally:
        warmup.cancel()
        if exporter is not None:
            await exporter.aclose()
        await app.state.qdrant.close()
        await app.state.http.aclose()
        await app.state.redis.aclose()


async def _warm_up(embedder: Embedder, reranker: Reranker) -> None:
    try:
        await embedder.embed(["warm-up"])
        log_event(logger, "embedder_ready", dense_dim=embedder.dense_dim)
    except Exception as exc:  # noqa: BLE001 - surfaced on the first real request too
        log_event(logger, "embedder_warmup_failed", level=logging.ERROR, error=str(exc))
    try:
        await reranker.rerank("warm-up", [_WARMUP_CHUNK], top_n=1)
        log_event(logger, "reranker_ready", reranker=reranker.name)
    except Exception as exc:  # noqa: BLE001 - surfaced on the first real request too
        log_event(logger, "reranker_warmup_failed", level=logging.ERROR, error=str(exc))


_WARMUP_CHUNK = RetrievedChunk(doc_id="warm-up", chunk_index=0, text="warm-up")


def get_pipeline(request: Request) -> RagPipeline:
    return request.app.state.pipeline


app = FastAPI(title="Enterprise RAG Gateway", version="0.1.0", lifespan=lifespan)
app.mount("/ui", StaticFiles(directory=WEB_DIR, html=True), name="ui")


@app.middleware("http")
async def trace_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    trace_id = start_trace()
    response = await call_next(request)
    response.headers["X-Trace-ID"] = trace_id
    return response


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/ui/")


@app.post("/dev/token", include_in_schema=False)
async def dev_token(
    body: DevTokenRequest, settings: Annotated[Settings, Depends(get_settings)]
) -> DevTokenResponse:
    """Mint a short-lived JWT for the demo UI. Exists only when RAG_ENV=dev.

    The token still goes through the normal `get_identity` verification on every request;
    this route only stands in for the upstream SSO that would issue it in production.
    """
    if settings.env != "dev":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not Found")
    if body.tenant_id not in settings.known_tenants:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "unknown tenant")
    log_event(
        logger,
        "dev_token_issued",
        tenant_id=body.tenant_id,
        role=body.role.value,
        user_id=body.user_id,
    )
    token = issue_token(
        tenant_id=body.tenant_id,
        role=body.role,
        user_id=body.user_id,
        settings=settings,
        ttl=DEV_TOKEN_TTL,
    )
    return DevTokenResponse(token=token, expires_in=int(DEV_TOKEN_TTL.total_seconds()))


@app.get("/metrics", include_in_schema=False)
async def metrics_endpoint() -> PlainTextResponse:
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness: the process is up. No dependency checks."""
    return {"status": "ok"}


@app.get("/ready")
async def ready(
    request: Request, settings: Annotated[Settings, Depends(get_settings)]
) -> JSONResponse:
    """Readiness: Qdrant reachable with the collection present, Redis answering PING."""
    checks: dict[str, str] = {}
    try:
        r = await request.app.state.http.get(f"{settings.qdrant_url}/readyz", timeout=2.0)
        checks["qdrant"] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
    except httpx.HTTPError as exc:
        checks["qdrant"] = type(exc).__name__
    if checks["qdrant"] == "ok":
        # Reachable but never ingested still cannot answer anything.
        try:
            exists = await request.app.state.qdrant.collection_exists(settings.qdrant_collection)
            checks["collection"] = "ok" if exists else "missing"
        except Exception as exc:  # noqa: BLE001
            checks["collection"] = type(exc).__name__
    try:
        await request.app.state.redis.ping()
        checks["redis"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["redis"] = type(exc).__name__
    ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        {"status": "ready" if ok else "not_ready", "checks": checks},
        status_code=status.HTTP_200_OK if ok else status.HTTP_503_SERVICE_UNAVAILABLE,
    )


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    identity: Annotated[Identity, Depends(get_identity)],
    pipeline: Annotated[RagPipeline, Depends(get_pipeline)],
) -> ChatResponse | JSONResponse | StreamingResponse:
    """`stream: false` -> one JSON ChatResponse. `stream: true` -> text/event-stream with
    `meta` (model chosen, sent once the first token exists), `delta` (answer text), then
    `done` (the full ChatResponse) or `error`. Failures before the first token are plain HTTP
    errors, exactly as in the non-streaming case."""
    risk = screen_text(body.query)
    log_event(
        logger,
        "chat_request",
        tenant_id=identity.tenant_id,
        role=identity.role.value,
        user_id=identity.user_id,
        query=body.query,  # masked by the JSON formatter
        query_risk=risk,
        turns=len(body.history),
        stream=body.stream,
    )
    if not body.stream:
        try:
            return await pipeline.answer(body, identity)
        except Exception as exc:
            return _error_response(exc)

    events = pipeline.stream(body, identity)
    try:
        first = await anext(events)
    except Exception as exc:
        return _error_response(exc)
    return PipelineStreamingResponse(
        _sse(first, events),
        events=events,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _sse(first: StreamEvent, events: AsyncIterator[StreamEvent]) -> AsyncIterator[str]:
    try:
        yield _sse_frame(first.kind, first.data)
        async for event in events:
            yield _sse_frame(event.kind, event.data)
    except Exception as exc:  # noqa: BLE001 - after the first token: report and end
        log_event(
            logger,
            "chat_stream_failed",
            level=logging.ERROR,
            error=type(exc).__name__,
            status_code=getattr(exc, "status_code", None),
        )
        yield _sse_frame(
            "error",
            {"trace_id": current_trace_id(), "detail": "the answer stream was interrupted"},
        )
    finally:
        await events.aclose()  # type: ignore[attr-defined]


class PipelineStreamingResponse(StreamingResponse):
    """Closes the pipeline's event generator however the response ends.

    On a client disconnect Starlette (ASGI spec < 2.4) cancels the response task group. If
    that happens before the body iterator starts (the client left while retrieval / rerank
    ran), `_sse` never runs and its `finally` never fires, leaving the pipeline generator
    suspended: no budget settle, no audit line, no rag_request_summary. Closing it here,
    shielded from the cancellation and time-bounded, guarantees that cleanup. Closing an
    already-finished generator is a no-op.
    """

    def __init__(
        self, content: AsyncIterator[str], *, events: AsyncIterator[StreamEvent], **kwargs: Any
    ) -> None:
        super().__init__(content, **kwargs)
        self._events = events

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.move_on_after(STREAM_CLEANUP_TIMEOUT_S, shield=True):
                await self._events.aclose()  # type: ignore[attr-defined]


def _sse_frame(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _error_response(exc: Exception) -> JSONResponse:
    """Map a pipeline failure to an HTTP error carrying the trace_id; re-raise the unknown."""
    if isinstance(exc, BudgetExceededError):
        log_event(logger, "chat_rejected", level=logging.WARNING, reason="budget")
        now = datetime.now(UTC)
        tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        response = _error(status.HTTP_429_TOO_MANY_REQUESTS, "the daily token budget is exhausted")
        response.headers["Retry-After"] = str(int((tomorrow - now).total_seconds()) + 1)
        return response
    if isinstance(exc, AllModelsFailedError):
        # Retries and the whole fallback chain are exhausted: a transient outage.
        log_event(logger, "chat_failed", level=logging.ERROR, stage="llm", error=type(exc).__name__)
        return _error(status.HTTP_503_SERVICE_UNAVAILABLE, "all language models are unavailable")
    if isinstance(exc, LLMError):
        # Non-retryable (bad request, auth, content refusal): no fallback was attempted.
        log_event(
            logger,
            "chat_failed",
            level=logging.ERROR,
            stage="llm",
            error=type(exc).__name__,
            status_code=exc.status_code,
        )
        return _error(status.HTTP_502_BAD_GATEWAY, "the language model call failed")
    if isinstance(exc, (ResponseHandlingException, UnexpectedResponse, httpx.HTTPError)):
        log_event(logger, "chat_failed", level=logging.ERROR, stage="retrieve", error=str(exc))
        return _error(status.HTTP_503_SERVICE_UNAVAILABLE, "the knowledge base is unavailable")
    if isinstance(exc, RedisError):
        # The budget cannot be checked: fail closed rather than spend without a limit.
        log_event(logger, "chat_failed", level=logging.ERROR, stage="redis", error=str(exc))
        return _error(status.HTTP_503_SERVICE_UNAVAILABLE, "the budget service is unavailable")
    raise exc


def _error(code: int, detail: str) -> JSONResponse:
    return JSONResponse({"trace_id": current_trace_id(), "detail": detail}, status_code=code)
