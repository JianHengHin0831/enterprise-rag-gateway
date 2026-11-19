"""Reranker (Top-20 fused hits -> Top-5 by cross-encoder score) + low-score gate.

Scores are in the reranker's native scale and the refusal threshold is calibrated per reranker
with the golden set (`python -m evals.sweep_threshold`). `BgeReranker` returns sigmoid-normalized
cross-encoder scores in [0, 1]; `FakeReranker` (tests) returns lexical overlap in [0, 1].
"""

import asyncio
import re
from typing import Any, Protocol

from app.config import Settings
from app.schemas import RetrievedChunk


class Reranker(Protocol):
    name: str

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]: ...


def passes_gate(ranked: list[RetrievedChunk], threshold: float) -> bool:
    """False => refuse without calling the LLM."""
    if not ranked:
        return False
    top = ranked[0].rerank_score
    return top is not None and top >= threshold


def apply_scores(
    chunks: list[RetrievedChunk], scores: list[float], top_n: int
) -> list[RetrievedChunk]:
    """Attach `rerank_score`, sort by it (stable: fusion order breaks ties), keep `top_n`."""
    if len(scores) != len(chunks):  # clearer message than zip(strict=True)
        raise ValueError(f"reranker returned {len(scores)} scores for {len(chunks)} chunks")
    scored = [
        c.model_copy(update={"rerank_score": float(s)}) for c, s in zip(chunks, scores, strict=True)
    ]
    scored.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
    return scored[:top_n]


# -------------------------------------------------------------------------- fake

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORD_TEXT = (
    "a an and are as at be by can do does for from how i in is it my of on or our the to "
    "we what when where which who why with you your"
)
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS}


class FakeReranker:
    """Deterministic, model-free: share of the query's content words found in the chunk."""

    name = "fake"

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        q = _terms(query)
        scores = [len(q & _terms(c.text)) / len(q) if q else 0.0 for c in chunks]
        return apply_scores(chunks, scores, top_n)


# -------------------------------------------------------------------------- bge-reranker


class BgeReranker:
    """BAAI/bge-reranker-v2-m3 cross-encoder via `transformers` (installed by `.[embed]`).

    Same scores as FlagEmbedding's `FlagReranker(normalize=True)` (sigmoid of the logit, in
    [0, 1]) at ~40% less CPU time. Loaded lazily; scoring is CPU/GPU bound, so it runs in a
    worker thread behind a lock (the model is not re-entrant). Pairs are batched shortest
    first to keep padding low.
    """

    name = "bge-reranker"

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        device: str | None = None,
        use_fp16: bool = False,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self._model_name = model_name
        self._device = device or "cpu"
        self._use_fp16 = use_fp16
        self._batch_size = batch_size
        self._max_length = max_length
        self._model: Any = None
        self._tokenizer: Any = None
        self._lock = asyncio.Lock()

    def _load(self) -> tuple[Any, Any]:
        if self._model is None:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - depends on optional extra
                raise RuntimeError(
                    'bge-reranker needs the optional extra: pip install -e ".[embed]" '
                    "(or set RAG_RERANKER=fake)"
                ) from exc
            model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
            if self._use_fp16:
                model = model.to(torch.float16)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = model.to(self._device).eval()
        return self._tokenizer, self._model

    def _score(self, query: str, texts: list[str]) -> list[float]:
        import torch

        tokenizer, model = self._load()
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        scores = [0.0] * len(texts)
        with torch.inference_mode():
            for b in range(0, len(order), self._batch_size):
                idx = order[b : b + self._batch_size]
                enc = tokenizer(
                    [[query, texts[i]] for i in idx],
                    padding=True,
                    truncation=True,
                    max_length=self._max_length,
                    return_tensors="pt",
                ).to(self._device)
                logits = model(**enc).logits.view(-1).float()
                for i, s in zip(idx, torch.sigmoid(logits).tolist(), strict=True):
                    scores[i] = s
        return scores

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        async with self._lock:
            scores = await asyncio.to_thread(self._score, query, [c.text for c in chunks])
        return apply_scores(chunks, scores, top_n)


def build_reranker(settings: Settings) -> Reranker:
    if settings.reranker == "fake":
        return FakeReranker()
    return BgeReranker(
        settings.rerank_model_name,
        device=settings.embed_device,
        use_fp16=settings.embed_use_fp16,
        batch_size=settings.embed_batch_size,
        max_length=settings.embed_max_length,
    )
