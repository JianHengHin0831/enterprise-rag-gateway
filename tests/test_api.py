import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


@pytest.fixture
def client(settings):
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health_has_trace_header(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert len(r.headers["X-Trace-ID"]) == 32


def test_chat_requires_bearer_token(client):
    assert client.post("/v1/chat", json={"query": "hi"}).status_code == 401


def test_role_header_is_ignored(client):
    """Spoofed identity headers must not authenticate anyone."""
    r = client.post(
        "/v1/chat",
        json={"query": "salary bands"},
        headers={"X-Tenant-ID": "acme_corp", "X-User-Role": "hr"},
    )
    assert r.status_code == 401


def test_chat_with_token_reaches_pipeline(client, make_token):
    r = client.post(
        "/v1/chat",
        json={"query": "expense process"},
        headers={"Authorization": f"Bearer {make_token()}"},
    )
    assert r.status_code == 501  # TODO(M1): expect 200 once the pipeline exists
    assert r.json()["trace_id"] == r.headers["X-Trace-ID"]
