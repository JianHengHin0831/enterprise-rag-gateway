import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from redis.asyncio import Redis

from app.config import Settings, get_settings
from app.observability.logging import configure_logging, log_event
from app.observability.trace import current_trace_id, start_trace
from app.schemas import ChatRequest, DevTokenRequest, DevTokenResponse, Identity
from app.security.auth import get_identity, issue_token
from app.security.injection import screen_text

logger = logging.getLogger("rag.api")

WEB_DIR = Path(__file__).parent / "web"
DEV_TOKEN_TTL = timedelta(hours=1)


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
