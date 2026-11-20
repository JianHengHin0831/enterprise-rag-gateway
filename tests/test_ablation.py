"""Pure metric functions of the retrieval ablation (`evals/ablation.py`)."""

import pytest

from app.schemas import RetrievedChunk, Role
from evals.ablation import (
    CONFIGS,
    _offline_client,
    best_gate_accuracy,
    roc_auc,
    run,
    score_case,
    summarize,
)
from evals.golden import GoldenCase, load_golden


def _case(**kw) -> GoldenCase:
    base = {
        "id": "T-01",
        "category": "answerable",
        "tenant_id": "acme_corp",
        "role": Role.ENG,
        "question": "q",
        "expected_keywords": ["alpha", ["beta", "b3ta"]],
        "expected_doc_ids": ["doc_a"],
        "expect_refusal": False,
        "reference_answer": "r",
    }
    return GoldenCase(**{**base, **kw})


def _chunk(doc: str, text: str, i: int = 0) -> RetrievedChunk:
    return RetrievedChunk(doc_id=doc, chunk_index=i, text=text)


def test_score_case_rank_recall_and_coverage():
    ctx = [_chunk("doc_x", "nothing"), _chunk("doc_a", "Alpha and B3TA here")]
    r = score_case(_case(), ctx, top1=0.9)
    assert r.doc_recall == 1.0
    assert r.rr == 0.5  # first expected doc at rank 2
    assert r.fact_coverage == 1.0  # case-insensitive, list = alternatives


def test_score_case_missing_doc_and_partial_facts():
    r = score_case(_case(), [_chunk("doc_x", "alpha only")], top1=0.1)
    assert (r.doc_recall, r.rr, r.fact_coverage) == (0.0, 0.0, 0.5)


def test_score_case_skips_refusal_and_calculation_cases():
    refuse = score_case(_case(expect_refusal=True, expected_keywords=[]), [], top1=None)
    assert (refuse.doc_recall, refuse.rr, refuse.fact_coverage) == (None, None, None)
    calc = score_case(_case(tags=["calculation"]), [_chunk("doc_a", "x")], top1=1.0)
    assert calc.doc_recall == 1.0 and calc.fact_coverage is None


def test_roc_auc():
    assert roc_auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert roc_auc([0.1], [0.9]) == 0.0
    assert roc_auc([0.5], [0.5]) == 0.5  # ties count half
    assert roc_auc([], [0.1]) is None


def test_best_gate_accuracy_respects_answer_recall_floor():
    answer = [0.9, 0.8, 0.7, 0.05]  # one should-answer case scores very low
    refuse = [0.1, 0.2]
    # Without the floor the best gate would block 0.05; with a 100% floor it must pass it.
    acc, t = best_gate_accuracy(answer, refuse, min_answer_recall=1.0)
    assert t == 0.05 and acc == pytest.approx(4 / 6)
    acc, t = best_gate_accuracy(answer, refuse, min_answer_recall=0.75)
    assert t == 0.7 and acc == pytest.approx(5 / 6)


def test_summarize_treats_empty_retrieval_as_refused():
    results = [
        score_case(_case(id="a"), [_chunk("doc_a", "alpha beta")], top1=0.9),
        score_case(_case(id="r", expect_refusal=True, expected_keywords=[]), [], top1=None),
    ]
    s = summarize(results)
    assert s["gate_auc"] == 1.0 and s["gate_accuracy"] == 1.0
    assert s["full_coverage"] == 1.0


async def test_offline_run_covers_every_config(settings):
    cases = load_golden()[:4]
    s = settings.model_copy(update={"qdrant_collection": "ablation_test"})
    client = await _offline_client(s)
    try:
        out = await run(s, cases, client)
    finally:
        await client.close()
    assert set(out) == {name for name, _, _ in CONFIGS}
    assert all(len(rs) == len(cases) for rs in out.values())
