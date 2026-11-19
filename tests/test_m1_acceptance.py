"""M1 acceptance, through the real HTTP route and the real pipeline.

intern and hr ask the same salary question: different trace_ids in the logs, and the intern's
`retrieved_doc_ids` contain no hr documents. In-memory Qdrant + FakeEmbedder + FakeLLM.
"""

import json
import logging

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.gateway.base import NonRetryableLLMError, RetryableLLMError
from app.gateway.providers.fake import FakeLLM
from app.gateway.providers.template import TemplateProvider
from app.gateway.router import ModelRouter, load_models_config
from app.main import app, get_pipeline
from app.observability.logging import JsonFormatter
from app.rag.embeddings import FakeEmbedder
from app.rag.pipeline import RagPipeline
from app.rag.prompts import load_prompt
from app.rag.retriever import HybridRetriever
from app.schemas import Role
from tests.conftest import TEST_COLLECTION

SALARY_Q = "What is the P6 salary band and how is severance calculated?"


class JsonLines(logging.Handler):
    """Formats at emit time, so trace_id comes from the live request context."""

    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(JsonFormatter())
        self.events: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.events.append(json.loads(self.format(record)))


def _pipeline(settings, qdrant, llm) -> RagPipeline:
    return RagPipeline(
        retriever=HybridRetriever(qdrant, FakeEmbedder(), TEST_COLLECTION),
        router=ModelRouter(
            config=load_models_config(settings.models_config_path),
            providers={"dashscope": llm, "local": TemplateProvider()},
        ),
        prompt=load_prompt(settings.prompts_dir, "rag_qa", 1),
        retrieve_top_k=20,
        context_top_n=5,
    )


@pytest.fixture
def run(settings, indexed_qdrant):
    """Yield (client, logs, set_llm) with the real pipeline behind /v1/chat."""
    state = {"llm": FakeLLM("dashscope", default="See the policy [1].")}
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_pipeline] = lambda: _pipeline(
        settings, indexed_qdrant, state["llm"]
    )
    logs = JsonLines()
    with TestClient(app) as client:
        logging.getLogger().addHandler(logs)  # after lifespan's configure_logging
        try:
            yield client, logs, state
        finally:
            logging.getLogger().removeHandler(logs)
    app.dependency_overrides.clear()


def _ask(client, make_token, role: Role, tenant="acme_corp", query=SALARY_Q):
    token = make_token(tenant=tenant, role=role, user=f"u_{role.value}")
    return client.post(
        "/v1/chat", json={"query": query}, headers={"Authorization": f"Bearer {token}"}
    )


def _retrieve_span(logs: JsonLines, trace_id: str) -> dict:
    [span] = [
        e
        for e in logs.events
        if e["event"] == "span" and e.get("span") == "retrieve" and e["trace_id"] == trace_id
    ]
    return span


def test_intern_and_hr_same_question_isolated_in_logs(run, make_token):
    client, logs, _ = run
    r_intern = _ask(client, make_token, Role.INTERN)
    r_hr = _ask(client, make_token, Role.HR)
    assert r_intern.status_code == r_hr.status_code == 200

    tid_intern, tid_hr = r_intern.json()["trace_id"], r_hr.json()["trace_id"]
    assert tid_intern and tid_hr and tid_intern != tid_hr
    assert tid_intern == r_intern.headers["X-Trace-ID"]

    intern_docs = _retrieve_span(logs, tid_intern)["retrieved_doc_ids"]
    hr_docs = _retrieve_span(logs, tid_hr)["retrieved_doc_ids"]
    assert intern_docs and not [d for d in intern_docs if d.startswith("acme_hr_")]
    assert "acme_hr_compensation" in hr_docs  # positive control

    # Every log line of a request carries that request's trace_id, and the route logged the
    # caller identity from the JWT.
    requests = [e for e in logs.events if e["event"] == "chat_request"]
    assert {(e["trace_id"], e["role"]) for e in requests} == {
        (tid_intern, "intern"),
        (tid_hr, "hr"),
    }


def test_beta_user_never_gets_acme_docs_end_to_end(run, make_token):
    client, logs, _ = run
    for role in Role:
        r = _ask(client, make_token, role, tenant="beta_inc")
        assert r.status_code == 200
        span = _retrieve_span(logs, r.json()["trace_id"])
        assert not [d for d in span["retrieved_doc_ids"] if not d.startswith("beta_")]
        assert all(c["doc_id"].startswith("beta_") for c in r.json()["citations"])


def test_response_shape(run, make_token):
    client, _, _ = run
    body = _ask(client, make_token, Role.HR).json()
    assert body["answer"] == "See the policy [1]."
    assert body["model_used"] == "qwen-plus" and body["degraded"] is False
    [citation] = body["citations"]
    assert citation["index"] == 1 and citation["doc_id"].startswith("acme_")


def test_llm_failure_returns_502_with_trace_id(run, make_token):
    # M2: a retryable 503 now falls back (see test below); a non-retryable error still fails.
    client, _, state = run
    state["llm"] = FakeLLM("dashscope", script=[NonRetryableLLMError("401", status_code=401)])
    r = _ask(client, make_token, Role.HR)
    assert r.status_code == 502
    assert r.json()["trace_id"] == r.headers["X-Trace-ID"]


def test_retryable_llm_failure_falls_back_to_template(run, make_token):
    client, _, state = run
    state["llm"] = FakeLLM("dashscope", script=[RetryableLLMError("503", status_code=503)] * 3)
    r = _ask(client, make_token, Role.HR)
    assert r.status_code == 200
    body = r.json()
    assert body["model_used"] == "fallback-template" and body["degraded"] is True
    assert len(state["llm"].calls) == 3  # retry.max_attempts in configs/models.yaml
