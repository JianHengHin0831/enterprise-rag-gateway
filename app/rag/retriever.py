"""ACL-enforced hybrid retrieval over Qdrant.

TODO(M1): implement with the Query API:

    acl = build_acl_filter(identity)
    client.query_points(
        collection_name=...,
        prefetch=[
            models.Prefetch(query=dense, using="dense", filter=acl, limit=top_k),
            models.Prefetch(query=sparse, using="sparse", filter=acl, limit=top_k),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        filter=acl,          # belt and braces
        limit=top_k,
        with_payload=True,
    )

The filter is put on EVERY prefetch explicitly; do not rely on top-level filter propagation.
`tests/test_retriever_acl.py` (M1) must assert that unauthorized chunks never come back.
"""

from qdrant_client import AsyncQdrantClient

from app.rag.embeddings import Embedder
from app.schemas import Identity, RetrievedChunk


class HybridRetriever:
    def __init__(self, client: AsyncQdrantClient, embedder: Embedder, collection: str) -> None:
        self._client = client
        self._embedder = embedder
        self._collection = collection

    async def search(self, query: str, identity: Identity, top_k: int) -> list[RetrievedChunk]:
        raise NotImplementedError("HybridRetriever.search — milestone M1")
