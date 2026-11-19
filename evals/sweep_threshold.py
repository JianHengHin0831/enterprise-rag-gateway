"""Calibrate the low-score refusal threshold on the golden set.

python -m evals.sweep_threshold                     # needs Qdrant with the corpus ingested
python -m evals.sweep_threshold --svg docs/threshold_curve.svg
python -m evals.sweep_threshold --from-report evals/reports/threshold_sweep.json  # re-sweep only

For every golden case: mint + verify a JWT for (tenant_id, role), run the real ACL-filtered
hybrid retrieval (Top-20) and the configured reranker (Top-5), and record the Top-1 rerank
score. No LLM is called: this measures the gate alone.

Gate labels come from `expect_refusal`: a case "should answer" when it is False, "should
refuse" when True. For each threshold t (gate passes when top-1 >= t):
- answer_recall  = share of should-answer cases that pass the gate (1 - false refusals)
- refusal_recall = share of should-refuse cases the gate refuses on its own
- gate_accuracy  = correct gate decisions / all cases
- refusal_rate   = share of all cases refused by the gate

Recommended threshold: the highest gate_accuracy among thresholds that keep answer_recall >=
--min-answer-recall (default 0.95: a blocked answer cannot be recovered, while weak context
that passes the gate is still refused by the LLM, so the gate is tuned to rarely block);
ties resolve to the middle of the widest best plateau, for margin on both sides.

Scoring runs the reranker on CPU and is slow; `--from-report` re-sweeps the saved scores.
"""

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from qdrant_client import AsyncQdrantClient

from app.config import Settings, get_settings
from app.rag.embeddings import build_embedder
from app.rag.reranker import build_reranker
from app.rag.retriever import HybridRetriever
from app.security.auth import decode_token, issue_token
from evals.run_eval import GoldenCase, load_golden

REPORT_PATH = Path(__file__).parent / "reports" / "threshold_sweep.json"
STEP = 0.01


@dataclass(frozen=True)
class CaseScore:
    id: str
    category: str
    should_refuse: bool
    top1: float | None  # None: nothing retrieved
    top_refs: list[str]
    expected_in_top_n: bool | None  # None: case has no expected_doc_ids


@dataclass(frozen=True)
class Point:
    threshold: float
    answer_recall: float
    refusal_recall: float
    gate_accuracy: float
    refusal_rate: float


async def score_cases(settings: Settings, cases: list[GoldenCase]) -> list[CaseScore]:
    client = AsyncQdrantClient(url=settings.qdrant_url, check_compatibility=False)
    retriever = HybridRetriever(client, build_embedder(settings), settings.qdrant_collection)
    reranker = build_reranker(settings)
    out: list[CaseScore] = []
    try:
        for c in cases:
            # Identity only from a verified JWT, exactly like the request path.
            token = issue_token(
                tenant_id=c.tenant_id,
                role=c.role,
                user_id=f"eval_{c.id}",
                settings=settings,
                ttl=timedelta(minutes=5),
            )
            identity = decode_token(token, settings)
            hits = await retriever.search(c.question, identity, settings.retrieve_top_k)
            ranked = await reranker.rerank(c.question, hits, settings.rerank_top_n)
            top_docs = {r.doc_id for r in ranked}
            out.append(
                CaseScore(
                    id=c.id,
                    category=c.category.value,
                    should_refuse=c.expect_refusal,
                    top1=ranked[0].rerank_score if ranked else None,
                    top_refs=[r.ref for r in ranked],
                    expected_in_top_n=(
                        bool(top_docs & set(c.expected_doc_ids)) if c.expected_doc_ids else None
                    ),
                )
            )
            print(f"  {c.id:<6} {c.category:<13} top1={_fmt(out[-1].top1)}", flush=True)
    finally:
        await client.close()
    return out


def sweep(scores: list[CaseScore], step: float = STEP) -> list[Point]:
    answer = [s for s in scores if not s.should_refuse]
    refuse = [s for s in scores if s.should_refuse]
    points = []
    for i in range(round(1 / step) + 1):
        t = round(i * step, 4)
        passed = {s.id for s in scores if s.top1 is not None and s.top1 >= t}
        a_ok = sum(s.id in passed for s in answer)
        r_ok = sum(s.id not in passed for s in refuse)
        points.append(
            Point(
                threshold=t,
                answer_recall=a_ok / len(answer) if answer else 1.0,
                refusal_recall=r_ok / len(refuse) if refuse else 1.0,
                gate_accuracy=(a_ok + r_ok) / len(scores),
                refusal_rate=(len(scores) - len(passed)) / len(scores),
            )
        )
    return points


def recommend(points: list[Point], min_answer_recall: float) -> Point:
    ok = [p for p in points if p.answer_recall >= min_answer_recall - 1e-9]
    if not ok:
        raise ValueError(f"no threshold keeps answer_recall >= {min_answer_recall}")
    best = max(p.gate_accuracy for p in ok)
    # Widest contiguous run of thresholds at the best accuracy; pick its middle.
    runs: list[list[Point]] = []
    for p in ok:
        if abs(p.gate_accuracy - best) > 1e-9:
            continue
        if runs and abs(runs[-1][-1].threshold + STEP - p.threshold) < 1e-6:
            runs[-1].append(p)
        else:
            runs.append([p])
    widest = max(runs, key=len)
    return widest[(len(widest) - 1) // 2]


def render_svg(points: list[Point], chosen: Point, scores: list[CaseScore]) -> str:
    """Dependency-free line chart: threshold (x) vs the four curves (y, 0..1)."""
    w, h, left, right, top, bottom = 720, 400, 56, 170, 24, 48
    pw, ph = w - left - right, h - top - bottom

    def x(t: float) -> float:  # sqrt scale: the interesting region is near 0
        return left + (t**0.5) * pw

    def y(v: float) -> float:
        return top + (1 - v) * ph

    series = [
        ("gate_accuracy", "Gate accuracy", "#2563eb", ""),
        ("answer_recall", "Answer recall", "#16a34a", ""),
        ("refusal_recall", "Refusal recall", "#dc2626", ""),
        ("refusal_rate", "Refusal rate (all)", "#6b7280", ' stroke-dasharray="5 4"'),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'font-family="sans-serif" font-size="12">',
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
    ]
    for i in range(11):
        v = i / 10
        parts.append(
            f'<line x1="{left}" x2="{left + pw}" y1="{y(v):.1f}" y2="{y(v):.1f}" '
            f'stroke="#e5e7eb"/>'
            f'<text x="{left - 8}" y="{y(v) + 4:.1f}" text-anchor="end" fill="#374151">'
            f"{v:.1f}</text>"
        )
    for v in (0.0, 0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
        parts.append(
            f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top}" y2="{top + ph}" '
            f'stroke="#f3f4f6"/>'
            f'<text x="{x(v):.1f}" y="{top + ph + 18}" text-anchor="middle" fill="#374151">'
            f"{v:g}</text>"
        )
    for s in scores:  # top-1 score ticks under the axis: green = should answer, red = refuse
        if s.top1 is not None:
            color = "#dc2626" if s.should_refuse else "#16a34a"
            parts.append(
                f'<line x1="{x(s.top1):.1f}" x2="{x(s.top1):.1f}" y1="{top + ph - 8}" '
                f'y2="{top + ph}" stroke="{color}" stroke-opacity="0.7"/>'
            )
    for key, _label, color, dash in series:
        pts = " ".join(f"{x(p.threshold):.1f},{y(getattr(p, key)):.1f}" for p in points)
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"{dash}/>'
        )
    cx = x(chosen.threshold)
    parts.append(
        f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{top}" y2="{top + ph}" stroke="#111827" '
        f'stroke-dasharray="2 3"/>'
        f'<text x="{cx + 4:.1f}" y="{top + 12}" fill="#111827">'
        f"chosen {chosen.threshold:.2f}</text>"
    )
    for i, (_key, label, color, dash) in enumerate(series):
        ly = top + 10 + i * 20
        parts.append(
            f'<line x1="{left + pw + 16}" x2="{left + pw + 40}" y1="{ly}" y2="{ly}" '
            f'stroke="{color}" stroke-width="2"{dash}/>'
            f'<text x="{left + pw + 46}" y="{ly + 4}" fill="#111827">{label}</text>'
        )
    parts.append(
        f'<text x="{left + pw / 2:.1f}" y="{h - 8}" text-anchor="middle" fill="#111827">'
        "Rerank threshold (top-1 score, sqrt scale)</text></svg>"
    )
    return "\n".join(parts) + "\n"


def _fmt(v: float | None) -> str:
    return "  none" if v is None else f"{v:.4f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-answer-recall", type=float, default=0.95)
    parser.add_argument(
        "--from-report", type=Path, help="reuse the case scores of an earlier report"
    )
    parser.add_argument("--svg", type=Path, help="also write the curve as an SVG chart")
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    settings = get_settings()
    if args.from_report:
        saved = json.loads(args.from_report.read_text(encoding="utf-8"))
        scores = [CaseScore(**c) for c in saved["cases"]]
        print(f"{len(scores)} cases from {args.from_report} ({saved['rerank_model_name']})")
    else:
        cases = load_golden()
        print(
            f"{len(cases)} cases | embedder={settings.embedder} reranker={settings.reranker} "
            f"({settings.rerank_model_name}) top_k={settings.retrieve_top_k} "
            f"top_n={settings.rerank_top_n} current threshold={settings.rerank_threshold}"
        )
        scores = asyncio.run(score_cases(settings, cases))
    points = sweep(scores)
    chosen = recommend(points, args.min_answer_recall)

    answerable = [s for s in scores if s.expected_in_top_n is not None]
    hit_rate = sum(bool(s.expected_in_top_n) for s in answerable) / max(len(answerable), 1)
    max_refuse = max((s.top1 or 0.0 for s in scores if s.should_refuse), default=0.0)
    min_answer = min((s.top1 or 0.0 for s in scores if not s.should_refuse), default=0.0)

    print("\nthreshold  answer_recall  refusal_recall  gate_accuracy  refusal_rate")
    for p in points:
        if p.threshold <= 0.1 or round(p.threshold * 100) % 10 == 0 or p is chosen:
            mark = "  <- chosen" if p is chosen else ""
            print(
                f"  {p.threshold:5.2f}    {p.answer_recall:9.1%}    {p.refusal_recall:10.1%}"
                f"    {p.gate_accuracy:9.1%}    {p.refusal_rate:9.1%}{mark}"
            )
    print(
        f"\nexpected doc in Top-{settings.rerank_top_n} after rerank: {hit_rate:.1%} "
        f"({len(answerable)} cases with expected_doc_ids)"
    )
    print(f"lowest top-1 among should-answer cases: {min_answer:.4f}")
    print(f"highest top-1 among should-refuse cases: {max_refuse:.4f}")
    print(
        f"recommended RAG_RERANK_THRESHOLD={chosen.threshold:.2f} "
        f"(min answer recall {args.min_answer_recall:.0%})"
    )

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "embedder": settings.embedder,
        "reranker": settings.reranker,
        "rerank_model_name": settings.rerank_model_name,
        "retrieve_top_k": settings.retrieve_top_k,
        "rerank_top_n": settings.rerank_top_n,
        "min_answer_recall": args.min_answer_recall,
        "recommended_threshold": chosen.threshold,
        "expected_doc_in_top_n_rate": hit_rate,
        "cases": [asdict(s) for s in scores],
        "curve": [asdict(p) for p in points],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lf = "\n"  # same bytes on every OS
    args.out.write_text(json.dumps(report, indent=2) + lf, encoding="utf-8", newline=lf)
    print(f"report: {args.out}")
    if args.svg:  # chart ticks: green = should answer, red = should refuse (top-1 score)
        args.svg.parent.mkdir(parents=True, exist_ok=True)
        args.svg.write_text(render_svg(points, chosen, scores), encoding="utf-8", newline=lf)
        print(f"chart:  {args.svg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
