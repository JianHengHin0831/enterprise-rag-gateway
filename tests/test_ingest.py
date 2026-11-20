"""Ingest: corpus shape, layout guards, and upsert into an in-memory Qdrant + fakeredis."""

from collections import Counter
from pathlib import Path

import pytest
from fakeredis import FakeAsyncRedis
from qdrant_client import AsyncQdrantClient, models

from app.cache.keys import index_version_key
from app.rag.embeddings import FakeEmbedder
from app.rag.ingest import Chunk, load_corpus, tenant_dirs, upsert
from app.rag.store import (
    DENSE_VECTOR,
    PAYLOAD_INDEXES,
    SPARSE_VECTOR,
    CollectionMismatchError,
    ensure_collection,
    point_id,
)

ROOT = Path(__file__).resolve().parents[1]
MOCK_DOCS = ROOT / "data" / "mock_docs"
COLLECTION = "test_chunks"
REQUIRED_PAYLOAD = {"tenant_id", "allowed_roles", "doc_id", "chunk_index", "index_version"}


# ----------------------------------------------------------------------------- corpus


@pytest.fixture(scope="module")
def corpus() -> list[Chunk]:
    return load_corpus(MOCK_DOCS, frozenset({"acme_corp", "beta_inc"}))


def test_corpus_size_and_coverage(corpus):
    docs = {(c.tenant_id, c.doc_id, tuple(c.allowed_roles)) for c in corpus}
    assert 15 <= len(docs) <= 20
    access = {(tenant, role) for tenant, _, roles in docs for role in roles}
    for tenant in ("acme_corp", "beta_inc"):
        for level in ("all", "eng", "hr"):
            assert (tenant, level) in access, f"missing {tenant}/{level}"


def test_two_malicious_docs_flagged_at_ingest(corpus):
    flagged_docs = {c.doc_id for c in corpus if c.suspicious_patterns}
    assert flagged_docs == {"acme_pantry_guide", "beta_it_faq"}


def test_chunk_payload_has_required_fields(corpus):
    payload = corpus[0].payload(index_version=3)
    assert payload.keys() >= REQUIRED_PAYLOAD
    assert payload["index_version"] == 3
    assert isinstance(payload["suspicious"], bool)


def _write_doc(root: Path, rel: str, *, doc_id: str, tenant: str, roles: str, body: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\ndoc_id: {doc_id}\ntenant_id: {tenant}\nallowed_roles: {roles}\n"
        f"title: {doc_id}\n---\n{body}\n",
        encoding="utf-8",
    )
    return path


def test_layout_tenant_mismatch_rejected(tmp_path):
    _write_doc(
        tmp_path, "acme_corp/all/x.md", doc_id="x", tenant="beta_inc", roles="[all]", body="text"
    )
    with pytest.raises(ValueError, match="does not match directory"):
        load_corpus(tmp_path)


def test_layout_access_mismatch_rejected(tmp_path):
    _write_doc(
        tmp_path, "acme_corp/hr/x.md", doc_id="x", tenant="acme_corp", roles="[all]", body="text"
    )
    with pytest.raises(ValueError, match="not in allowed_roles"):
        load_corpus(tmp_path)


def test_duplicate_doc_id_and_unknown_tenant_rejected(tmp_path):
    _write_doc(
        tmp_path, "acme_corp/all/a.md", doc_id="dup", tenant="acme_corp", roles="[all]", body="a"
    )
    _write_doc(
        tmp_path, "acme_corp/all/b.md", doc_id="dup", tenant="acme_corp", roles="[all]", body="b"
    )
    with pytest.raises(ValueError, match="duplicate doc_id"):
        load_corpus(tmp_path)
    with pytest.raises(ValueError, match="unknown tenant"):
        load_corpus(tmp_path, frozenset({"beta_inc"}))


# ----------------------------------------------------------------------------- upsert


@pytest.fixture
async def qdrant():
    client = AsyncQdrantClient(location=":memory:")
    yield client
    await client.close()


@pytest.fixture
def redis():
    return FakeAsyncRedis()


async def _all_points(client: AsyncQdrantClient) -> list[models.Record]:
    points, _ = await client.scroll(COLLECTION, limit=10_000, with_payload=True, with_vectors=True)
    return points


async def test_upsert_full_corpus(qdrant, redis, corpus):
    versions = await upsert(
        corpus, client=qdrant, embedder=FakeEmbedder(), redis=redis, collection=COLLECTION
    )
    assert versions == {"acme_corp": 1, "beta_inc": 1}
    assert int(await redis.get(index_version_key("acme_corp"))) == 1

    info = await qdrant.get_collection(COLLECTION)
    assert set(info.config.params.vectors) == {DENSE_VECTOR}
    assert info.config.params.vectors[DENSE_VECTOR].size == 64
    assert info.config.params.vectors[DENSE_VECTOR].distance == models.Distance.COSINE
    assert set(info.config.params.sparse_vectors) == {SPARSE_VECTOR}

    points = await _all_points(qdrant)
    assert len(points) == len(corpus)
    for p in points:
        assert p.payload.keys() >= REQUIRED_PAYLOAD
        assert p.id == point_id(
            p.payload["tenant_id"], p.payload["doc_id"], p.payload["chunk_index"]
        )
        assert set(p.vector) == {DENSE_VECTOR, SPARSE_VECTOR}
        assert p.payload["index_version"] == 1
    suspicious = Counter(p.payload["doc_id"] for p in points if p.payload["suspicious"])
    assert set(suspicious) == {"acme_pantry_guide", "beta_it_faq"}


async def test_reingest_is_idempotent_and_bumps_version(qdrant, redis, corpus):
    kwargs = {
        "client": qdrant,
        "embedder": FakeEmbedder(),
        "redis": redis,
        "collection": COLLECTION,
    }
    await upsert(corpus, **kwargs)
    versions = await upsert(corpus, **kwargs)
    assert versions == {"acme_corp": 2, "beta_inc": 2}
    points = await _all_points(qdrant)
    assert len(points) == len(corpus)
    assert {p.payload["index_version"] for p in points} == {2}


async def test_reingest_removes_stale_chunks_of_shrunk_doc(qdrant, redis, tmp_path):
    long_body = "\n\n".join(f"Paragraph {i}. " + "word " * 120 for i in range(4))
    path = _write_doc(
        tmp_path,
        "acme_corp/all/doc.md",
        doc_id="doc",
        tenant="acme_corp",
        roles="[all]",
        body=long_body,
    )
    _write_doc(
        tmp_path,
        "acme_corp/hr/other.md",
        doc_id="other",
        tenant="acme_corp",
        roles="[hr]",
        body="Other document.",
    )
    kwargs = {
        "client": qdrant,
        "embedder": FakeEmbedder(),
        "redis": redis,
        "collection": COLLECTION,
    }
    await upsert(load_corpus(tmp_path), **kwargs)
    before = await _all_points(qdrant)
    assert sum(p.payload["doc_id"] == "doc" for p in before) > 1

    # Re-ingest only the shrunk doc: its stale chunks go, the untouched doc stays.
    path.write_text(
        path.read_text(encoding="utf-8").split("Paragraph 0")[0] + "Short now.", encoding="utf-8"
    )
    (tmp_path / "acme_corp" / "hr" / "other.md").unlink()
    await upsert(load_corpus(tmp_path), **kwargs)
    after = await _all_points(qdrant)
    assert [p.payload["doc_id"] for p in after].count("doc") == 1
    assert any(p.payload["doc_id"] == "other" for p in after)


async def test_prune_deletes_docs_removed_from_source(qdrant, redis, tmp_path):
    for tenant, doc_id, access in [
        ("acme_corp", "keep", "all"),
        ("acme_corp", "gone", "hr"),
        ("beta_inc", "beta_doc", "all"),
    ]:
        _write_doc(
            tmp_path,
            f"{tenant}/{access}/{doc_id}.md",
            doc_id=doc_id,
            tenant=tenant,
            roles=f"[{access}]",
            body=f"Body of {doc_id}.",
        )
    kwargs = {
        "client": qdrant,
        "embedder": FakeEmbedder(),
        "redis": redis,
        "collection": COLLECTION,
    }
    await upsert(load_corpus(tmp_path), **kwargs)

    # Delete one acme doc; re-ingest everything, pruning acme only.
    (tmp_path / "acme_corp" / "hr" / "gone.md").unlink()
    versions = await upsert(
        load_corpus(tmp_path), **kwargs, prune_tenants=tenant_dirs(tmp_path) - {"beta_inc"}
    )
    docs = {(p.payload["tenant_id"], p.payload["doc_id"]) for p in await _all_points(qdrant)}
    assert docs == {("acme_corp", "keep"), ("beta_inc", "beta_doc")}
    # Caches of the pruned tenant are invalidated.
    assert int(await redis.get(index_version_key("acme_corp"))) == versions["acme_corp"] == 2


async def test_prune_tenant_with_no_docs_left_empties_it(qdrant, redis, tmp_path):
    path = _write_doc(
        tmp_path,
        "acme_corp/all/only.md",
        doc_id="only",
        tenant="acme_corp",
        roles="[all]",
        body="The only doc.",
    )
    kwargs = {
        "client": qdrant,
        "embedder": FakeEmbedder(),
        "redis": redis,
        "collection": COLLECTION,
    }
    await upsert(load_corpus(tmp_path), **kwargs)
    path.unlink()
    assert tenant_dirs(tmp_path) == {"acme_corp"}

    versions = await upsert(load_corpus(tmp_path), **kwargs, prune_tenants={"acme_corp"})
    assert await _all_points(qdrant) == []
    assert versions == {"acme_corp": 2}


async def test_prune_never_touches_tenants_outside_its_scope(qdrant, redis, tmp_path):
    for tenant in ("acme_corp", "beta_inc"):
        _write_doc(
            tmp_path,
            f"{tenant}/all/{tenant}_doc.md",
            doc_id=f"{tenant}_doc",
            tenant=tenant,
            roles="[all]",
            body="Body.",
        )
    kwargs = {
        "client": qdrant,
        "embedder": FakeEmbedder(),
        "redis": redis,
        "collection": COLLECTION,
    }
    await upsert(load_corpus(tmp_path), **kwargs)

    # Ingest only acme's directory with pruning: beta's points and version stay as they were.
    versions = await upsert(
        load_corpus(tmp_path / "acme_corp"), **kwargs, prune_tenants={"acme_corp"}
    )
    tenants = sorted(p.payload["tenant_id"] for p in await _all_points(qdrant))
    assert tenants == ["acme_corp", "beta_inc"]
    assert "beta_inc" not in versions
    assert int(await redis.get(index_version_key("beta_inc"))) == 1


async def test_ensure_collection_creates_acl_payload_indexes(qdrant, monkeypatch):
    calls: dict[str, object] = {}
    original = qdrant.create_payload_index

    async def spy(*, collection_name, field_name, field_schema, **kw):
        calls[field_name] = field_schema
        return await original(
            collection_name=collection_name, field_name=field_name, field_schema=field_schema, **kw
        )

    monkeypatch.setattr(qdrant, "create_payload_index", spy)
    await ensure_collection(qdrant, COLLECTION, dense_dim=8)
    assert {"tenant_id", "allowed_roles"} <= calls.keys()
    assert calls == PAYLOAD_INDEXES
    tenant_schema = calls["tenant_id"]
    assert isinstance(tenant_schema, models.KeywordIndexParams) and tenant_schema.is_tenant
    assert calls["allowed_roles"] == models.PayloadSchemaType.KEYWORD


async def test_ensure_collection_rejects_dim_mismatch_unless_recreate(qdrant):
    await ensure_collection(qdrant, COLLECTION, dense_dim=8)
    with pytest.raises(CollectionMismatchError, match="dense size 8"):
        await ensure_collection(qdrant, COLLECTION, dense_dim=16)
    await ensure_collection(qdrant, COLLECTION, dense_dim=16, recreate=True)
    info = await qdrant.get_collection(COLLECTION)
    assert info.config.params.vectors[DENSE_VECTOR].size == 16


async def test_hybrid_query_with_acl_filter_on_ingested_points(qdrant, redis, corpus):
    """Smoke check that the stored layout supports the M1 retriever's query shape."""
    from app.schemas import Identity, Role
    from app.security.acl import build_acl_filter

    embedder = FakeEmbedder()
    await upsert(corpus, client=qdrant, embedder=embedder, redis=redis, collection=COLLECTION)
    [q] = await embedder.embed(["salary band severance N+1"])
    intern = Identity(tenant_id="acme_corp", role=Role.INTERN, user_id="u")
    acl = build_acl_filter(intern)
    res = await qdrant.query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=q.dense, using=DENSE_VECTOR, filter=acl, limit=20),
            models.Prefetch(
                query=models.SparseVector(indices=q.sparse.indices, values=q.sparse.values),
                using=SPARSE_VECTOR,
                filter=acl,
                limit=20,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=acl,
        limit=20,
        with_payload=True,
    )
    assert res.points
    for p in res.points:
        assert p.payload["tenant_id"] == "acme_corp"
        assert set(p.payload["allowed_roles"]) & {"all", "intern"}
        assert p.payload["doc_id"] != "acme_hr_compensation"
