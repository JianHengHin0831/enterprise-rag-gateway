"""Ingest Markdown docs with front matter into Qdrant.

Doc format (data/mock_docs/<tenant>/<access>/*.md):

    ---
    doc_id: acme_handbook
    tenant_id: acme_corp
    allowed_roles: [all]
    title: Employee Handbook
    ---
    # body...

The directory layout is enforced: `<tenant>` must equal `tenant_id` and `<access>` must be one
of `allowed_roles`, so a copy-pasted front matter cannot silently publish a doc to another
tenant or role.

Pipeline: parse -> chunk -> `screen_text` (mark `suspicious`) -> embed (dense + sparse) ->
upsert with a deterministic point id -> delete stale chunks of re-ingested docs ->
bump `index_version:{tenant_id}` in Redis (last, so caches switch only once data is in place).

Usage:
    python -m app.rag.ingest data/mock_docs              # embed + upsert
    python -m app.rag.ingest data/mock_docs --dry-run    # parse/chunk/screen only
    python -m app.rag.ingest data/mock_docs --recreate   # drop and rebuild the collection
"""

import argparse
import asyncio
import logging
import re
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from qdrant_client import AsyncQdrantClient, models
from redis.asyncio import Redis

from app.cache.keys import index_version_key
from app.config import get_settings
from app.observability.logging import configure_logging, log_event
from app.rag.embeddings import Embedder, build_embedder
from app.rag.store import DENSE_VECTOR, SPARSE_VECTOR, ensure_collection, point_id
from app.schemas import PUBLIC_ACCESS, Role
from app.security.injection import screen_text

logger = logging.getLogger("rag.ingest")

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_VALID_ACCESS = {PUBLIC_ACCESS, *(r.value for r in Role)}


@dataclass
class SourceDoc:
    doc_id: str
    tenant_id: str
    allowed_roles: list[str]
    title: str
    body: str
    path: Path


@dataclass
class Chunk:
    doc_id: str
    tenant_id: str
    allowed_roles: list[str]
    title: str
    chunk_index: int
    text: str
    suspicious_patterns: list[str] = field(default_factory=list)

    def payload(self, index_version: int) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "tenant_id": self.tenant_id,
            "allowed_roles": self.allowed_roles,
            "title": self.title,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "suspicious": bool(self.suspicious_patterns),
            "suspicious_patterns": self.suspicious_patterns,
            "index_version": index_version,
        }

    def embed_text(self) -> str:
        """Text sent to the embedder: the title gives short chunks their document context."""
        return f"{self.title}\n\n{self.text}"


def parse_doc(path: Path) -> SourceDoc:
    raw = path.read_text(encoding="utf-8")
    m = _FRONT_MATTER.match(raw)
    if not m:
        raise ValueError(f"{path}: missing front matter")
    meta = yaml.safe_load(m.group(1)) or {}
    for key in ("doc_id", "tenant_id", "allowed_roles"):
        if key not in meta:
            raise ValueError(f"{path}: front matter missing '{key}'")
    roles = list(meta["allowed_roles"])
    if not roles or not set(roles) <= _VALID_ACCESS:
        raise ValueError(f"{path}: invalid allowed_roles {roles}")
    return SourceDoc(
        doc_id=meta["doc_id"],
        tenant_id=meta["tenant_id"],
        allowed_roles=roles,
        title=meta.get("title", meta["doc_id"]),
        body=raw[m.end() :],
        path=path,
    )


def check_layout(doc: SourceDoc) -> None:
    """Front matter must agree with the `<tenant>/<access>/` directory the file lives in."""
    access_dir = doc.path.parent.name
    tenant_dir = doc.path.parent.parent.name
    if doc.tenant_id != tenant_dir:
        raise ValueError(
            f"{doc.path}: tenant_id '{doc.tenant_id}' does not match directory '{tenant_dir}'"
        )
    if access_dir not in doc.allowed_roles:
        raise ValueError(
            f"{doc.path}: directory '{access_dir}' not in allowed_roles {doc.allowed_roles}"
        )


def chunk_doc(doc: SourceDoc, max_chars: int = 500, overlap: int = 80) -> list[Chunk]:
    """Split by Markdown headings/paragraphs, then pack into ~max_chars windows.

    TODO(M3): tune max_chars/overlap via eval; consider heading-path prefixes per chunk.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.body) if p.strip()]
    windows: list[str] = []
    buf = ""
    for p in paragraphs:
        if buf and len(buf) + len(p) + 2 > max_chars:
            windows.append(buf)
            buf = buf[-overlap:] + "\n\n" + p if overlap else p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        windows.append(buf)
    return [
        Chunk(
            doc_id=doc.doc_id,
            tenant_id=doc.tenant_id,
            allowed_roles=doc.allowed_roles,
            title=doc.title,
            chunk_index=i,
            text=text,
            suspicious_patterns=screen_text(text),
        )
        for i, text in enumerate(windows)
    ]


def load_corpus(root: Path, known_tenants: frozenset[str] | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: dict[str, Path] = {}
    for path in sorted(root.rglob("*.md")):
        doc = parse_doc(path)
        check_layout(doc)
        if known_tenants is not None and doc.tenant_id not in known_tenants:
            raise ValueError(f"{path}: unknown tenant '{doc.tenant_id}'")
        if doc.doc_id in seen:
            raise ValueError(
                f"{path}: duplicate doc_id '{doc.doc_id}' (also in {seen[doc.doc_id]})"
            )
        seen[doc.doc_id] = path
        chunks.extend(chunk_doc(doc))
    return chunks


def _batched(items: Sequence[Chunk], size: int) -> Iterator[Sequence[Chunk]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


async def upsert(
    chunks: list[Chunk],
    *,
    client: AsyncQdrantClient,
    embedder: Embedder,
    redis: Redis,
    collection: str,
    batch_size: int = 32,
    recreate: bool = False,
) -> dict[str, int]:
    """Embed and upsert chunks; return the new index_version per ingested tenant."""
    if not chunks:
        return {}
    await ensure_collection(client, collection, embedder.dense_dim, recreate=recreate)

    tenants = sorted({c.tenant_id for c in chunks})
    versions: dict[str, int] = {}
    for tenant in tenants:
        current = await redis.get(index_version_key(tenant))
        versions[tenant] = int(current or 0) + 1

    for batch in _batched(chunks, batch_size):
        embeddings = await embedder.embed([c.embed_text() for c in batch])
        points = [
            models.PointStruct(
                id=point_id(c.tenant_id, c.doc_id, c.chunk_index),
                vector={
                    DENSE_VECTOR: e.dense,
                    SPARSE_VECTOR: models.SparseVector(
                        indices=e.sparse.indices, values=e.sparse.values
                    ),
                },
                payload=c.payload(versions[c.tenant_id]),
            )
            for c, e in zip(batch, embeddings, strict=True)
        ]
        await client.upsert(collection_name=collection, points=points, wait=True)

    # A doc that shrank leaves chunks with the old version behind; drop them.
    for tenant in tenants:
        doc_ids = sorted({c.doc_id for c in chunks if c.tenant_id == tenant})
        await client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="tenant_id", match=models.MatchValue(value=tenant)
                        ),
                        models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids)),
                    ],
                    must_not=[
                        models.FieldCondition(
                            key="index_version", match=models.MatchValue(value=versions[tenant])
                        )
                    ],
                )
            ),
            wait=True,
        )

    for tenant, version in versions.items():
        await redis.set(index_version_key(tenant), version)
    return versions


async def _run(root: Path, *, recreate: bool) -> dict[str, int]:
    settings = get_settings()
    chunks = load_corpus(root, settings.known_tenants)
    client = AsyncQdrantClient(url=settings.qdrant_url)
    redis = Redis.from_url(settings.redis_url)
    try:
        return await upsert(
            chunks,
            client=client,
            embedder=build_embedder(settings),
            redis=redis,
            collection=settings.qdrant_collection,
            batch_size=settings.embed_batch_size,
            recreate=recreate,
        )
    finally:
        await client.close()
        await redis.aclose()


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="python -m app.rag.ingest")
    parser.add_argument("root", nargs="?", default="data/mock_docs", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="parse, chunk and screen only")
    parser.add_argument("--recreate", action="store_true", help="drop and recreate the collection")
    args = parser.parse_args(argv[1:])

    configure_logging()
    settings = get_settings()
    chunks = load_corpus(args.root, settings.known_tenants)
    flagged = [c for c in chunks if c.suspicious_patterns]
    docs = {c.doc_id for c in chunks}
    print(f"parsed {len(docs)} docs, {len(chunks)} chunks; {len(flagged)} flagged suspicious")
    for c in flagged:
        print(f"  suspicious: {c.doc_id}#{c.chunk_index} {c.suspicious_patterns}")
    if args.dry_run:
        return

    print(f"embedding with '{settings.embedder}' and upserting into '{settings.qdrant_collection}'")
    versions = asyncio.run(_run(args.root, recreate=args.recreate))
    log_event(
        logger,
        "ingest_complete",
        collection=settings.qdrant_collection,
        chunks=len(chunks),
        suspicious=len(flagged),
        index_versions=versions,
    )
    print(f"done; index_version per tenant: {versions}")


if __name__ == "__main__":
    main(sys.argv)
