"""/ready reports Qdrant (reachable + collection present) and Redis, with fakes only."""

import httpx
import pytest
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient

from app.config import get_settings
from app.main import app
from app.rag.store import ensure_collection
from tests.conftest import TEST_COLLECTION


class BrokenRedis:
    async def ping(self):
        raise ConnectionError("down")

    async def aclose(self):
        pass


@pytest.fixture
def client(settings):
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _wire(client, *, qdrant_status: int | None, qdrant, redis) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if qdrant_status is None:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(qdrant_status)

    client.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client.app.state.qdrant = qdrant
    client.app.state.redis = redis


def test_ready_when_everything_is_up(client):
    qdrant = AsyncQdrantClient(location=":memory:")
    client.portal.call(ensure_collection, qdrant, TEST_COLLECTION, 8)
    _wire(client, qdrant_status=200, qdrant=qdrant, redis=FakeAsyncRedis())
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ready",
        "checks": {"qdrant": "ok", "collection": "ok", "redis": "ok"},
    }


def test_not_ready_before_ingest(client):
    _wire(
        client,
        qdrant_status=200,
        qdrant=AsyncQdrantClient(location=":memory:"),
        redis=FakeAsyncRedis(),
    )
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["checks"]["collection"] == "missing"


def test_not_ready_when_qdrant_down(client):
    _wire(
        client,
        qdrant_status=None,
        qdrant=AsyncQdrantClient(location=":memory:"),
        redis=FakeAsyncRedis(),
    )
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["checks"]["qdrant"] == "ConnectError"
    assert "collection" not in r.json()["checks"]


def test_not_ready_when_redis_down(client):
    qdrant = AsyncQdrantClient(location=":memory:")
    client.portal.call(ensure_collection, qdrant, TEST_COLLECTION, 8)
    _wire(client, qdrant_status=200, qdrant=qdrant, redis=BrokenRedis())
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json()["checks"]["redis"] == "ConnectionError"
