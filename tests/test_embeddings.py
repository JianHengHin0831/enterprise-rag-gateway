import math

import numpy as np
import pytest

from app.config import Settings
from app.rag.embeddings import (
    BgeM3Embedder,
    FakeEmbedder,
    build_embedder,
    convert_bge_m3_output,
    to_sparse,
)


async def test_fake_embedder_shapes_and_normalization():
    emb = FakeEmbedder(dense_dim=32)
    [e] = await emb.embed(["Annual leave policy for interns"])
    assert len(e.dense) == 32
    assert math.isclose(sum(v * v for v in e.dense), 1.0, rel_tol=1e-9)
    assert e.sparse.indices == sorted(set(e.sparse.indices))
    assert len(e.sparse.indices) == len(e.sparse.values) == 5
    assert all(v > 0 for v in e.sparse.values)


async def test_fake_embedder_is_deterministic_across_instances():
    a = await FakeEmbedder().embed(["salary band P6"])
    b = await FakeEmbedder().embed(["salary band P6"])
    assert a == b


async def test_fake_embedder_similarity_tracks_word_overlap():
    q, near, far = await FakeEmbedder(dense_dim=256).embed(
        ["expense approval limit", "expense approval limit for managers", "pantry coffee machine"]
    )

    def cos(x: list[float], y: list[float]) -> float:
        return sum(i * j for i, j in zip(x, y, strict=True))

    assert cos(q.dense, near.dense) > cos(q.dense, far.dense)
    assert set(q.sparse.indices) <= set(near.sparse.indices)
    assert not set(q.sparse.indices) & set(far.sparse.indices)


async def test_fake_embedder_empty_text_has_valid_dense():
    [e] = await FakeEmbedder().embed(["   "])
    assert any(e.dense) and e.sparse.indices == []


def test_to_sparse_sorts_drops_nonpositive_and_merges():
    sv = to_sparse({"42": 0.3, "7": 0.9, "13": 0.0, 42: 0.5, "99": -0.1})
    assert sv.indices == [7, 42]
    assert sv.values == [0.9, 0.5]


def test_convert_bge_m3_output():
    dense = np.array([[0.6, 0.8], [1.0, 0.0]], dtype=np.float32)
    lexical = [{"2009": 0.25, "100": 0.1}, {}]
    a, b = convert_bge_m3_output(dense, lexical)
    assert a.dense == pytest.approx([0.6, 0.8]) and isinstance(a.dense[0], float)
    assert a.sparse.indices == [100, 2009] and a.sparse.values == pytest.approx([0.1, 0.25])
    assert b.sparse.indices == []


def test_convert_bge_m3_output_length_mismatch():
    with pytest.raises(ValueError):
        convert_bge_m3_output([[1.0]], [{}, {}])


def test_build_embedder_selects_backend():
    fake = build_embedder(Settings(_env_file=None, embedder="fake", fake_embed_dim=16))
    assert isinstance(fake, FakeEmbedder) and fake.dense_dim == 16
    real = build_embedder(Settings(_env_file=None, embedder="bge-m3"))
    assert isinstance(real, BgeM3Embedder) and real.dense_dim == 1024


async def test_bge_m3_embedder_encodes_via_model(monkeypatch):
    """No model download: stub the loaded model and check the call contract."""

    class StubModel:
        calls: list[dict] = []

        def encode(self, texts, **kwargs):
            self.calls.append(kwargs)
            return {
                "dense_vecs": np.ones((len(texts), 1024), dtype=np.float32),
                "lexical_weights": [{"5": 0.5}] * len(texts),
            }

    emb = BgeM3Embedder(batch_size=4)
    stub = StubModel()
    monkeypatch.setattr(emb, "_load", lambda: stub)
    out = await emb.embed(["a", "b"])
    assert len(out) == 2 and len(out[0].dense) == 1024
    assert stub.calls[0]["return_dense"] and stub.calls[0]["return_sparse"]
    assert not stub.calls[0]["return_colbert_vecs"]
    assert await emb.embed([]) == []
