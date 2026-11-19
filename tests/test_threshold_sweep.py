"""Threshold sweep math (evals/sweep_threshold.py) on hand-made scores; no model, no Qdrant."""

import pytest

from evals.sweep_threshold import CaseScore, recommend, render_svg, sweep


def _case(cid: str, top1: float | None, should_refuse: bool) -> CaseScore:
    return CaseScore(
        id=cid,
        category="no_answer" if should_refuse else "answerable",
        should_refuse=should_refuse,
        top1=top1,
        top_refs=[],
        expected_in_top_n=None,
    )


SCORES = [
    _case("a1", 0.90, False),
    _case("a2", 0.40, False),
    _case("a3", 0.20, False),
    _case("r1", 0.05, True),
    _case("r2", None, True),  # nothing retrieved: refused at any threshold
]


def _at(points, t):
    return next(p for p in points if abs(p.threshold - t) < 1e-9)


def test_sweep_counts_gate_decisions():
    points = sweep(SCORES)
    p0 = _at(points, 0.0)
    assert p0.answer_recall == 1.0 and p0.refusal_recall == 0.5  # r2 never passes
    p = _at(points, 0.10)
    assert p.answer_recall == 1.0 and p.refusal_recall == 1.0 and p.gate_accuracy == 1.0
    assert p.refusal_rate == pytest.approx(2 / 5)
    p = _at(points, 0.30)
    assert p.answer_recall == pytest.approx(2 / 3) and p.gate_accuracy == pytest.approx(4 / 5)
    assert _at(points, 1.0).answer_recall == 0.0


def test_recommend_picks_middle_of_best_plateau():
    # Perfect accuracy for t in (0.05, 0.20] -> thresholds 0.06..0.20 (15 points), middle 0.13.
    chosen = recommend(sweep(SCORES), min_answer_recall=1.0)
    assert chosen.gate_accuracy == 1.0
    assert chosen.threshold == pytest.approx(0.13)


def test_recommend_respects_min_answer_recall():
    scores = [*SCORES, _case("a4", 0.01, False)]  # an answerable case below every refusal
    assert recommend(sweep(scores), min_answer_recall=1.0).threshold <= 0.01
    relaxed = recommend(sweep(scores), min_answer_recall=0.75)
    assert relaxed.answer_recall >= 0.75 and relaxed.threshold > 0.05


def test_svg_renders_and_marks_the_choice():
    points = sweep(SCORES)
    svg = render_svg(points, recommend(points, 1.0), SCORES)
    assert svg.startswith("<svg") and "chosen 0.13" in svg
