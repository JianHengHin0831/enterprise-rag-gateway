from datetime import timedelta
from pathlib import Path

import pytest
from fakeredis import FakeAsyncRedis
from pydantic import SecretStr
from qdrant_client import AsyncQdrantClient

from app.config import Settings
from app.rag.embeddings import FakeEmbedder
from app.rag.ingest import Chunk, load_corpus, upsert
from app.schemas import Identity, Role
from app.security.auth import issue_token

ROOT = Path(__file__).resolve().parents[1]
MOCK_DOCS = ROOT / "data" / "mock_docs"
TEST_COLLECTION = "test_chunks"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        jwt_secret=SecretStr("test-secret-at-least-32-bytes-long!!"),
        embedder="fake",  # never download a model in tests
        qdrant_collection=TEST_COLLECTION,
    )


@pytest.fixture
def make_token(settings: Settings):
    def _make(
        tenant: str = "acme_corp",
        role: Role = Role.INTERN,
        user: str = "u1",
        ttl: timedelta = timedelta(minutes=5),
    ) -> str:
        return issue_token(tenant_id=tenant, role=role, user_id=user, settings=settings, ttl=ttl)

    return _make


@pytest.fixture
def intern() -> Identity:
    return Identity(tenant_id="acme_corp", role=Role.INTERN, user_id="u_intern")


@pytest.fixture
def hr() -> Identity:
    return Identity(tenant_id="acme_corp", role=Role.HR, user_id="u_hr")


@pytest.fixture(scope="session")
def corpus() -> list[Chunk]:
    return load_corpus(MOCK_DOCS, frozenset({"acme_corp", "beta_inc"}))


@pytest.fixture(scope="session")
def doc_meta(corpus) -> dict[str, tuple[str, frozenset[str]]]:
    """doc_id -> (tenant_id, allowed_roles), straight from the source files."""
    return {c.doc_id: (c.tenant_id, frozenset(c.allowed_roles)) for c in corpus}


@pytest.fixture
async def indexed_qdrant(corpus):
    """In-memory Qdrant holding the whole mock corpus, embedded with FakeEmbedder."""
    client = AsyncQdrantClient(location=":memory:")
    await upsert(
        corpus,
        client=client,
        embedder=FakeEmbedder(),
        redis=FakeAsyncRedis(),
        collection=TEST_COLLECTION,
    )
    yield client
    await client.close()
