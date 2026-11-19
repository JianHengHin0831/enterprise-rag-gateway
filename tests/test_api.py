import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app, get_pipeline
from app.observability.trace import current_trace_id
from app.schemas import ChatRequest, ChatResponse, Identity


class StubPipeline:
    """Echoes the verified identity back, so tests can see what the route passed in."""

    def __init__(self) -> None:
        self.calls: list[tuple[ChatRequest, Identity]] = []

    async def answer(self, request: ChatRequest, identity: Identity) -> ChatResponse:
        self.calls.append((request, identity))
        return ChatResponse(
            trace_id=current_trace_id() or "",
            answer=f"{identity.tenant_id}/{identity.role.value}",
            model_used="stub",
        )


@pytest.fixture
def pipeline() -> StubPipeline:
    return StubPipeline()


@pytest.fixture
def client(settings, pipeline):
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_pipeline] = lambda: pipeline
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


def test_chat_with_token_reaches_pipeline(client, make_token, pipeline):
    r = client.post(
        "/v1/chat",
        json={"query": "expense process"},
        headers={"Authorization": f"Bearer {make_token()}"},
    )
    assert r.status_code == 200
    assert r.json()["trace_id"] == r.headers["X-Trace-ID"]
    [(request, identity)] = pipeline.calls
    assert request.query == "expense process"
    assert (identity.tenant_id, identity.role.value) == ("acme_corp", "intern")


def test_identity_in_body_is_ignored(client, make_token, pipeline):
    """Only the JWT decides identity; extra body fields cannot override it."""
    r = client.post(
        "/v1/chat",
        json={"query": "salary", "tenant_id": "beta_inc", "role": "hr"},
        headers={"Authorization": f"Bearer {make_token()}"},
    )
    assert r.status_code == 200 and r.json()["answer"] == "acme_corp/intern"


def test_ui_is_served(client):
    r = client.get("/", follow_redirects=True)
    assert r.status_code == 200
    assert "RAG Gateway Console" in r.text


def test_dev_token_is_verified_like_any_other(client):
    r = client.post(
        "/dev/token", json={"tenant_id": "beta_inc", "role": "eng", "user_id": "u_eng_7"}
    )
    assert r.status_code == 200
    token = r.json()["token"]
    chat = client.post(
        "/v1/chat", json={"query": "hi"}, headers={"Authorization": f"Bearer {token}"}
    )
    assert chat.status_code == 200
    assert chat.json()["answer"] == "beta_inc/eng"


def test_dev_token_rejects_unknown_tenant(client):
    r = client.post("/dev/token", json={"tenant_id": "evil", "role": "hr", "user_id": "u1"})
    assert r.status_code == 400


def test_dev_token_disabled_outside_dev(settings):
    prod = settings.model_copy(update={"env": "prod"})
    app.dependency_overrides[get_settings] = lambda: prod
    try:
        with TestClient(app) as c:
            r = c.post("/dev/token", json={"tenant_id": "acme_corp", "role": "hr", "user_id": "u1"})
        assert r.status_code == 404
    finally:
        app.dependency_overrides.clear()
