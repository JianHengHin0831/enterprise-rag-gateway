"""Reranker interface + low-score gate.

TODO(M3): implement an API reranker (Jina / Cohere / SiliconFlow bge-reranker) via the shared
httpx client, or a local cross-encoder. Scores must be in the reranker's native scale; the
threshold is calibrated per reranker with the golden set.
"""

from typing import Protocol

from app.schemas import RetrievedChunk


class Reranker(Protocol):
    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]: ...


def passes_gate(ranked: list[RetrievedChunk], threshold: float) -> bool:
    """False => refuse without calling the LLM."""
    if not ranked:
        return False
    top = ranked[0].rerank_score
    return top is not None and top >= threshold
