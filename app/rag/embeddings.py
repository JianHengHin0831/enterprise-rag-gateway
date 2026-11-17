"""Embedding interface. Recommended: bge-m3 (dense + sparse in one model, strong on Chinese).

TODO(M1): implement one Embedder (local FlagEmbedding BGEM3FlagModel, or an API) and a
deterministic FakeEmbedder for tests.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class Embedding:
    dense: list[float]
    sparse: SparseVector


class Embedder(Protocol):
    dense_dim: int

    async def embed(self, texts: list[str]) -> list[Embedding]: ...
