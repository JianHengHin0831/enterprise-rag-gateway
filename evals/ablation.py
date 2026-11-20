"""Retrieval ablation: which part of dense + sparse + RRF + rerank earns its place?

python -m evals.ablation                       # needs Qdrant with the corpus ingested + bge models
python -m evals.ablation --offline             # in-memory Qdrant + fakes (harness smoke, seconds)

For every golden case the question is embedded once and three ACL-filtered Qdrant queries are
run, each with a candidate pool of `retrieve_top_k` (20): dense only, sparse only, and the
production hybrid (both prefetches + RRF). Each pool is used as is (first `rerank_top_n` = 8
hits) and after the cross-encoder reranker (pool -> Top-8), giving six configurations. The
Top-8 is exactly what the answering model would see. No LLM is called.

Context metrics (should-answer cases with `expected_doc_ids`):
- doc_recall      share of expected docs found in the Top-8 (mean over cases)
- mrr             1 / rank of the first expected doc in the Top-8 (0 when absent)
- fact_coverage   share of expected keywords present in the Top-8 text. A keyword missing
                  from the context cannot appear in a grounded answer, so this is the ceiling
                  on answer success. Cases tagged `calculation` are skipped (computed results).
- full_coverage   share of cases where EVERY expected keyword is in the Top-8

Gate metrics (all cases; "should refuse" = `expect_refusal`): can this configuration's top-1
score drive the low-score refusal gate?
- gate_auc        ROC AUC of top-1 score, should-answer vs should-refuse (0.5 = no signal)
- gate_accuracy   best gate accuracy over thresholds that keep answer recall >= 95% (the same
                  rule as `evals.sweep_threshold`)

The ACL filter from `build_acl_filter` is on every query, so the ablation never sees
unauthorized chunks either.
"""

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from app.config import Settings, get_settings
from app.rag.embeddings import build_embedder
from app.rag.reranker import build_reranker
from app.rag.retriever import build_hybrid_query
from app.rag.store import DENSE_VECTOR, SPARSE_VECTOR
from app.schemas import RetrievedChunk
from app.security.acl import build_acl_filter
from app.security.auth import decode_token, issue_token
from evals.golden import CORPUS_ROOT, UNGROUNDED_TAGS, GoldenCase, keyword_matches, load_golden
from evals.golden import normalize_for_match as norm

REPORT_DIR = Path(__file__).parent / "reports"
MIN_ANSWER_RECALL = 0.95
POOLS = ("dense", "sparse", "hybrid")
CONFIGS = [  # (name, pool, rerank)
    ("dense", "dense", False),
    ("sparse", "sparse", False),
    ("hybrid (RRF)", "hybrid", False),
    ("dense + rerank", "dense", True),
    ("sparse + rerank", "sparse", True),
    ("hybrid + rerank (prod)", "hybrid", True),
]


# ------------------------------------------------------------------------- metrics (pure)


@dataclass(frozen=True)
class CaseResult:
    id: str
    should_refuse: bool
    top1: float | None  # None: nothing retrieved
    doc_recall: float | None  # None: no expected_doc_ids or a should-refuse case
    rr: float | None
    fact_coverage: float | None  # None: no grounded keywords to check
    refs: list[str] = field(default_factory=list)


def score_case(case: GoldenCase, context: list[RetrievedChunk], top1: float | None) -> CaseResult:
    answerable = not case.expect_refusal and bool(case.expected_doc_ids)
    doc_recall = rr = coverage = None
    if answerable:
        docs = [c.doc_id for c in context]
        expected = set(case.expected_doc_ids)
        doc_recall = len(expected & set(docs)) / len(expected)
        rank = next((i for i, d in enumerate(docs, start=1) if d in expected), None)
        rr = 1.0 / rank if rank else 0.0
        if case.expected_keywords and not UNGROUNDED_TAGS & set(case.tags):
            text = norm("\n".join(c.text for c in context))
            hits = sum(keyword_matches(k, text) for k in case.expected_keywords)
            coverage = hits / len(case.expected_keywords)
    return CaseResult(
        id=case.id,
        should_refuse=case.expect_refusal,
        top1=top1,
        doc_recall=doc_recall,
        rr=rr,
        fact_coverage=coverage,
        refs=[c.ref for c in context],
    )


def roc_auc(positives: list[float], negatives: list[float]) -> float | None:
    """P(score of a random positive > a random negative), ties count half."""
    if not positives or not negatives:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def best_gate_accuracy(
    answer_scores: list[float], refuse_scores: list[float], min_answer_recall: float
) -> tuple[float | None, float | None]:
    """(accuracy, threshold) of the best gate "answer iff top1 >= t" that keeps answer recall
    >= min_answer_recall. Candidate thresholds are the observed scores (plus +inf)."""
    total = len(answer_scores) + len(refuse_scores)
    if not total:
        return None, None
    best: tuple[float, float] | None = None
    for t in sorted(set(answer_scores + refuse_scores)) + [float("inf")]:
        passed = sum(s >= t for s in answer_scores)
        if answer_scores and passed / len(answer_scores) < min_answer_recall:
            continue
        acc = (passed + sum(s < t for s in refuse_scores)) / total
        if best is None or acc > best[0]:
            best = (acc, t)
    return best if best else (None, None)


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    # A case with nothing retrieved is refused by any gate: score it below every real score.
    answer = [
        r.top1 if r.top1 is not None else float("-inf") for r in results if not r.should_refuse
    ]
    refuse = [r.top1 if r.top1 is not None else float("-inf") for r in results if r.should_refuse]
    coverage = [r.fact_coverage for r in results if r.fact_coverage is not None]
    acc, threshold = best_gate_accuracy(answer, refuse, MIN_ANSWER_RECALL)
    return {
        "answerable_cases": sum(r.doc_recall is not None for r in results),
        "coverage_cases": len(coverage),
        "doc_recall": _mean([r.doc_recall for r in results]),
        "mrr": _mean([r.rr for r in results]),
        "fact_coverage": _mean(coverage),
        "full_coverage": (sum(c == 1.0 for c in coverage) / len(coverage)) if coverage else None,
        "gate_auc": roc_auc(answer, refuse),
        "gate_accuracy": acc,
        "gate_threshold": threshold,
    }


# ------------------------------------------------------------------------- retrieval


async def query_pool(
    client: AsyncQdrantClient,
    collection: str,
    pool: str,
    dense: list[float],
    sparse: models.SparseVector,
    acl: models.Filter,
    top_k: int,
) -> list[RetrievedChunk]:
    if pool == "hybrid":
        kwargs = build_hybrid_query(dense, sparse, acl, top_k)
    else:
        kwargs = {
            "query": dense if pool == "dense" else sparse,
            "using": DENSE_VECTOR if pool == "dense" else SPARSE_VECTOR,
            "query_filter": acl,
            "limit": top_k,
            "with_payload": True,
            "with_vectors": False,
        }
    result = await client.query_points(collection_name=collection, **kwargs)
    return [
        RetrievedChunk(
            doc_id=p.payload["doc_id"],
            chunk_index=p.payload["chunk_index"],
            text=p.payload["text"],
            title=p.payload.get("title"),
            score=p.score,
        )
        for p in result.points
        if p.payload is not None
    ]


async def run(
    settings: Settings, cases: list[GoldenCase], client: AsyncQdrantClient
) -> dict[str, list[CaseResult]]:
    embedder = build_embedder(settings)
    reranker = build_reranker(settings)
    top_k, top_n = settings.retrieve_top_k, settings.rerank_top_n
    out: dict[str, list[CaseResult]] = {name: [] for name, _, _ in CONFIGS}
    for i, case in enumerate(cases, start=1):
        token = issue_token(
            tenant_id=case.tenant_id,
            role=case.role,
            user_id=f"ablation_{case.id}",
            settings=settings,
            ttl=timedelta(minutes=5),
        )
        identity = decode_token(token, settings)  # identity only from a verified JWT
        acl = build_acl_filter(identity)
        [emb] = await embedder.embed([case.question])
        sparse = models.SparseVector(indices=emb.sparse.indices, values=emb.sparse.values)
        pools = {
            p: await query_pool(
                client, settings.qdrant_collection, p, emb.dense, sparse, acl, top_k
            )
            for p in POOLS
        }
        reranked = {p: await reranker.rerank(case.question, pools[p], top_n) for p in POOLS}
        for name, pool, rerank in CONFIGS:
            if rerank:
                ctx = reranked[pool]
                top1 = ctx[0].rerank_score if ctx else None
            else:
                ctx = pools[pool][:top_n]
                top1 = ctx[0].score if ctx else None
            out[name].append(score_case(case, ctx, top1))
        print(f"\r{i}/{len(cases)} cases", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return out


async def _offline_client(settings: Settings) -> AsyncQdrantClient:
    from fakeredis import FakeAsyncRedis

    from app.rag.ingest import load_corpus, upsert

    client = AsyncQdrantClient(location=":memory:")
    await upsert(
        load_corpus(CORPUS_ROOT),
        client=client,
        embedder=build_embedder(settings),
        redis=FakeAsyncRedis(),
        collection=settings.qdrant_collection,
    )
    return client


# ------------------------------------------------------------------------- report


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:.1%}"


def _num(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


def render_table(summary: dict[str, dict[str, Any]], top_k: int, top_n: int) -> str:
    lines = [
        f"| Config (pool Top-{top_k} -> context Top-{top_n}) | Doc recall | MRR "
        "| Fact coverage | All facts in context | Gate AUC | Best gate accuracy |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, _, _ in CONFIGS:
        s = summary[name]
        lines.append(
            f"| {name} | {_pct(s['doc_recall'])} | {_num(s['mrr'])} | {_pct(s['fact_coverage'])} "
            f"| {_pct(s['full_coverage'])} | {_num(s['gate_auc'])} | {_pct(s['gate_accuracy'])} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--offline", action="store_true", help="in-memory Qdrant + fakes")
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="context size to evaluate (default: RAG_RERANK_TOP_N). The mock corpus is small, "
        "so a tight context (e.g. 3) shows ranking differences a Top-8 context hides.",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.offline:
        settings = settings.model_copy(
            update={"embedder": "fake", "reranker": "fake", "qdrant_collection": "ablation"}
        )
    if args.top_n is not None:
        settings = settings.model_copy(update={"rerank_top_n": args.top_n})
    cases = load_golden()
    print(
        f"{len(cases)} cases | embedder={settings.embedder} reranker={settings.reranker} "
        f"top_k={settings.retrieve_top_k} top_n={settings.rerank_top_n}",
        file=sys.stderr,
    )

    async def _go() -> dict[str, list[CaseResult]]:
        client = (
            await _offline_client(settings)
            if args.offline
            else AsyncQdrantClient(url=settings.qdrant_url, check_compatibility=False)
        )
        try:
            return await run(settings, cases, client)
        finally:
            await client.close()

    results = asyncio.run(_go())
    summary = {name: summarize(rs) for name, rs in results.items()}
    print(render_table(summary, settings.retrieve_top_k, settings.rerank_top_n))
    any_s = next(iter(summary.values()))
    print(
        f"\n{any_s['answerable_cases']} should-answer cases with expected docs, "
        f"{any_s['coverage_cases']} with checkable facts; gate over {len(cases)} cases."
    )

    REPORT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    mode = "_offline" if args.offline else ""
    path = REPORT_DIR / f"ablation{mode}_top{settings.rerank_top_n}_{stamp}.json"
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "offline": args.offline,
        "config": {
            "embedder": settings.embedder,
            "reranker": settings.reranker,
            "rerank_model_name": settings.rerank_model_name,
            "retrieve_top_k": settings.retrieve_top_k,
            "rerank_top_n": settings.rerank_top_n,
        },
        "summary": summary,
        "cases": {name: [asdict(r) for r in rs] for name, rs in results.items()},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"report: {path.relative_to(REPORT_DIR.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
