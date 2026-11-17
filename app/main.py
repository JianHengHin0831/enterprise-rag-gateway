import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from redis.asyncio import Redis

from app.config import get_settings
from app.observability.logging import configure_logging, log_event
from app.observability.trace import current_trace_id, start_trace
from app.schemas import ChatRequest, Identity
from app.security.auth import get_identity
from app.security.injection import screen_text

logger = logging.getLogger("rag.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings = get_settings()
    app.state.http = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=5.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
    app.state.redis = Redis.from_url(settings.redis_url)
    # TODO(M1): build Qdrant client, embedder, retriever, router, pipeline here.
    try:
        yield
    finally:
        await app.state.http.aclose()
        await app.state.redis.aclose()


app = FastAPI(title="Enterprise RAG Gateway", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def trace_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    trace_id = start_trace()
    response = await call_next(request)
    response.headers["X-Trace-ID"] = trace_id
    return response


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness: the process is up. No dependency checks."""
    return {"status": "ok"}


@app.get("/ready")
async def ready(request: Request) -> JSONResponse:
    """Readiness: dependencies reachable."""
    settings = get_settings()
    checks: dict[str, str] = {}
    try:
        r = await request.app.state.http.get(f"{settings.qdrant_url}/readyz", timeout=2.0)
        checks["qdrant"] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
    except httpx.HTTPError as exc:
        checks["qdrant"] = type(exc).__name__
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


@app.post("/v1/chat")
async def chat(
    body: ChatRequest, identity: Annotated[Identity, Depends(get_identity)]
) -> JSONResponse:
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
    )
    # TODO(M1): return await pipeline.answer(body, identity)
    return JSONResponse(
        {"trace_id": current_trace_id(), "detail": "pipeline not implemented (M1)"},
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
    )
