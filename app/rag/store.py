"""Qdrant collection layout shared by ingest and retrieval.

- Named vectors: `dense` (cosine) and `sparse` (bge-m3 lexical weights, dot product).
- Payload indexes: `tenant_id` (keyword, `is_tenant=True` so Qdrant co-locates each tenant's
  points) and `allowed_roles` (keyword) back the ACL filter; `doc_id` (keyword) and
  `index_version` (integer) back re-ingest cleanup.
"""

import uuid

from qdrant_client import AsyncQdrantClient, models

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

# field name -> index schema. tenant_id / allowed_roles are required by the ACL filter.
PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType | models.KeywordIndexParams] = {
    "tenant_id": models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True),
    "allowed_roles": models.PayloadSchemaType.KEYWORD,
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "index_version": models.PayloadSchemaType.INTEGER,
}

_POINT_NAMESPACE = uuid.UUID("6f1c2a9e-3b7d-4c55-9a51-2d8e0f4b7c10")


def point_id(tenant_id: str, doc_id: str, chunk_index: int) -> str:
    """Deterministic id, so re-ingesting a chunk overwrites it instead of duplicating it."""
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{tenant_id}/{doc_id}/{chunk_index}"))


class CollectionMismatchError(RuntimeError):
    pass


async def ensure_collection(
    client: AsyncQdrantClient, name: str, dense_dim: int, *, recreate: bool = False
) -> None:
    """Create the collection with named dense + sparse vectors and the payload indexes.

    Idempotent. An existing collection whose layout does not match is an error (switching
    embedders changes the dense size) — re-run with `recreate=True`.
    """
    exists = await client.collection_exists(name)
    if exists and recreate:
        await client.delete_collection(name)
        exists = False

    if exists:
        info = await client.get_collection(name)
        vectors = info.config.params.vectors
        sparse = info.config.params.sparse_vectors or {}
        dense = vectors.get(DENSE_VECTOR) if isinstance(vectors, dict) else None
        if dense is None or SPARSE_VECTOR not in sparse:
            raise CollectionMismatchError(
                f"collection '{name}' lacks named vectors '{DENSE_VECTOR}'/'{SPARSE_VECTOR}'; "
                "re-ingest with --recreate"
            )
        if dense.size != dense_dim:
            raise CollectionMismatchError(
                f"collection '{name}' has dense size {dense.size}, embedder produces "
                f"{dense_dim}; re-ingest with --recreate"
            )
    else:
        await client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams()},
        )

    # Creating an index that already exists is a no-op in Qdrant.
    for field, schema in PAYLOAD_INDEXES.items():
        await client.create_payload_index(
            collection_name=name, field_name=field, field_schema=schema, wait=True
        )
