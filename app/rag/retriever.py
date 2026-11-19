"""ACL-enforced hybrid retrieval over Qdrant (Query API).

    acl = build_acl_filter(identity)
    client.query_points(
        prefetch=[
            Prefetch(query=dense,  using="dense",  filter=acl, limit=top_k),
            Prefetch(query=sparse, using="sparse", filter=acl, limit=top_k),
        ],
        query=FusionQuery(fusion=RRF),
        query_filter=acl,      # belt and braces
        limit=top_k,
    )

The filter is put on EVERY prefetch explicitly; top-level filter propagation is not relied
on. Unauthorized chunks are never fetched, so nothing is post-filtered in Python.
"""

from qdrant_client import AsyncQdrantClient, models

from app.rag.embeddings import Embedder
from app.rag.store import DENSE_VECTOR, SPARSE_VECTOR
from app.schemas import Identity, RetrievedChunk
from app.security.acl import build_acl_filter


def build_hybrid_query(
    dense: list[float], sparse: models.SparseVector, acl: models.Filter, top_k: int
) -> dict:
    """Keyword arguments for `query_points` (kept separate so tests can inspect the shape)."""
    return {
        "prefetch": [
            models.Prefetch(query=dense, using=DENSE_VECTOR, filter=acl, limit=top_k),
            models.Prefetch(query=sparse, using=SPARSE_VECTOR, filter=acl, limit=top_k),
        ],
        "query": models.FusionQuery(fusion=models.Fusion.RRF),
        "query_filter": acl,
        "limit": top_k,
        "with_payload": True,
        "with_vectors": False,
    }


class HybridRetriever:
    def __init__(self, client: AsyncQdrantClient, embedder: Embedder, collection: str) -> None:
        self._client = client
        self._embedder = embedder
        self._collection = collection

    async def search(self, query: str, identity: Identity, top_k: int) -> list[RetrievedChunk]:
        acl = build_acl_filter(identity)
        [emb] = await self._embedder.embed([query])
        sparse = models.SparseVector(indices=emb.sparse.indices, values=emb.sparse.values)
        result = await self._client.query_points(
            collection_name=self._collection,
            **build_hybrid_query(emb.dense, sparse, acl, top_k),
        )
        return [
            RetrievedChunk(
                doc_id=p.payload["doc_id"],
                chunk_index=p.payload["chunk_index"],
                text=p.payload["text"],
                title=p.payload.get("title"),
                score=p.score,
                suspicious=bool(p.payload.get("suspicious", False)),
            )
            for p in result.points
            if p.payload is not None
        ]
