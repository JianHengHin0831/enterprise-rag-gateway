"""Embedding interface: one call returns a dense AND a sparse vector per text.

Backends (`RAG_EMBEDDER`):
- `bge-m3`: BAAI/bge-m3 via FlagEmbedding (local, CPU or CUDA). Dense = 1024-d normalized
  CLS embedding, sparse = learned lexical weights keyed by tokenizer id. Install with
  `pip install -e ".[embed]"`.
- `fake`: deterministic hashing embedder. No model download; used by tests and for wiring
  smoke tests. Retrieval quality is lexical only.

Queries and documents are embedded the same way (bge-m3 uses no query instruction).
"""

import asyncio
import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from app.config import Settings


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


def to_sparse(weights: Mapping[Any, float], *, min_weight: float = 0.0) -> SparseVector:
    """Convert `{token_id: weight}` into a SparseVector with sorted, unique int indices.

    Qdrant rejects duplicate indices; zero/negative weights carry no signal and are dropped.
    """
    merged: dict[int, float] = {}
    for key, weight in weights.items():
        w = float(weight)
        if w <= min_weight:
            continue
        idx = int(key)
        merged[idx] = max(merged.get(idx, 0.0), w)
    indices = sorted(merged)
    return SparseVector(indices=indices, values=[merged[i] for i in indices])


def _l2_normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return [0.0] * len(vec)
    return [v / norm for v in vec]


# ---------------------------------------------------------------------------- fake

_TOKEN = re.compile(r"[a-z0-9]+")


def _stable_hash(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


class FakeEmbedder:
    """Deterministic, dependency-free embedder (feature hashing over lowercase word tokens).

    Texts sharing words get similar dense vectors and overlapping sparse indices, which is
    enough to exercise hybrid retrieval and ACL filtering without a model.
    """

    def __init__(self, dense_dim: int = 64) -> None:
        self.dense_dim = dense_dim

    def _embed_one(self, text: str) -> Embedding:
        counts = Counter(_TOKEN.findall(text.lower()))
        dense = [0.0] * self.dense_dim
        sparse: dict[int, float] = {}
        for token, tf in counts.items():
            h = _stable_hash(token)
            sign = 1.0 if (h >> 63) & 1 else -1.0
            dense[h % self.dense_dim] += sign * tf
            sparse[h % (2**31 - 1)] = 1.0 + math.log(tf)
        if not counts:  # an all-zero dense vector is invalid for cosine distance
            dense[0] = 1.0
        return Embedding(dense=_l2_normalize(dense), sparse=to_sparse(sparse))

    async def embed(self, texts: list[str]) -> list[Embedding]:
        return [self._embed_one(t) for t in texts]


# -------------------------------------------------------------------------- bge-m3


class BgeM3Embedder:
    """BAAI/bge-m3 dense + sparse via FlagEmbedding.

    The model is loaded lazily on first use. Encoding is CPU/GPU bound, so it runs in a worker
    thread; a lock serializes calls because the underlying model is not re-entrant.
    """

    dense_dim = 1024

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        device: str | None = None,
        use_fp16: bool = False,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._use_fp16 = use_fp16
        self._batch_size = batch_size
        self._max_length = max_length
        self._model: Any = None
        self._lock = asyncio.Lock()

    def _load(self) -> Any:
        if self._model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise RuntimeError(
                    'bge-m3 needs the optional extra: pip install -e ".[embed]" '
                    "(or set RAG_EMBEDDER=fake)"
                ) from exc
            kwargs: dict[str, Any] = {"use_fp16": self._use_fp16}
            if self._device:
                kwargs["devices"] = self._device
            self._model = BGEM3FlagModel(self._model_name, **kwargs)
        return self._model

    def _encode(self, texts: list[str]) -> list[Embedding]:
        out = self._load().encode(
            texts,
            batch_size=self._batch_size,
            max_length=self._max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return convert_bge_m3_output(out["dense_vecs"], out["lexical_weights"])

    async def embed(self, texts: list[str]) -> list[Embedding]:
        if not texts:
            return []
        async with self._lock:
            return await asyncio.to_thread(self._encode, texts)


def convert_bge_m3_output(
    dense_vecs: Iterable[Sequence[float]], lexical_weights: Iterable[Mapping[Any, float]]
) -> list[Embedding]:
    """Map FlagEmbedding's raw output (numpy rows + `{token_id_str: weight}`) to Embeddings."""
    return [
        Embedding(dense=[float(x) for x in dense], sparse=to_sparse(weights))
        for dense, weights in zip(dense_vecs, lexical_weights, strict=True)
    ]


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "fake":
        return FakeEmbedder(dense_dim=settings.fake_embed_dim)
    return BgeM3Embedder(
        settings.embed_model_name,
        device=settings.embed_device,
        use_fp16=settings.embed_use_fp16,
        batch_size=settings.embed_batch_size,
        max_length=settings.embed_max_length,
    )
