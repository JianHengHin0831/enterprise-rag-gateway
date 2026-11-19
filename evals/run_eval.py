"""Golden-set evaluation, end to end through the real `/v1/chat` route.

python -m evals.run_eval                    # full eval, compared with evals/baselines/latest.json
python -m evals.run_eval --judge            # + grounded LLM-as-judge faithfulness (extra calls)
python -m evals.run_eval --dry-run          # validate the golden set only
python -m evals.run_eval --offline          # no Qdrant / Redis / models / LLM: harness smoke run
python -m evals.run_eval --save-baseline    # only when explicitly intended

How a case runs (live mode): mint a JWT for (tenant_id, role) and POST the question to
`/v1/chat` in-process (httpx ASGITransport + the app lifespan), so auth, query screening, the
pipeline, the cache, the budget and the audit log all run exactly as in production. The
retrieval-level leak check reads `retrieved_doc_ids` from the request's audit line (matched
by trace_id), and document visibility is decided from the source files.

Judge (`--judge`): the rerank span's `context_refs` (the exact Top-N chunks the answering model
saw, in prompt order) are captured from the request's span log by trace_id, the chunks are read
back from Qdrant, wrapped with `wrap_documents` and given to the judge with the question and the
answer (`configs/prompts/judge_v2.yaml`). Faithfulness therefore means "every claim is supported
by what was retrieved", so correct detail beyond the one-line `reference_answer` is not
penalised. Answered (not refused) cases with captured contexts are judged; cache hits have none.

Isolation: the eval uses its own Redis database (`--redis-db`, default 15), flushed at start;
the tenants' `index_version` keys are copied from the main database so cache keys stay valid.
Every run therefore starts with an empty cache and a fresh daily budget, and never touches
the dev cache or budget. The audit log goes to a temporary file, not `logs/audit.jsonl`.

Scoring and the hard gates are in `evals/scoring.py`. Exit code: 0 = all hard gates hold
(no leaks, no successful injections, no errors), 1 = a hard gate failed, 2 = setup problem.
"""

import argparse
import asyncio
import json
import logging
import re
import sys
import tempfile
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.gateway.base import ChatMessage
from app.gateway.router import ModelRouter
from app.observability.logging import _FIELDS_ATTR
from app.observability.trace import current_trace_id
from app.rag.prompts import load_prompt
from app.schemas import RetrievedChunk
from app.security.auth import issue_token
from app.security.injection import wrap_documents
from evals.golden import (  # noqa: F401 - re-exported for callers of evals.run_eval
    CORPUS_ROOT,
    GOLDEN_PATH,
    TARGET_MIX,
    Category,
    GoldenCase,
    can_access,
    category_mix,
    load_corpus_docs,
    load_golden,
    normalize_for_match,
)
from evals.scoring import (
    CaseOutcome,
    CaseResult,
    compare,
    hard_failures,
    render_table,
    result_dicts,
    score_case,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = Path(__file__).parent / "baselines" / "latest.json"
REPORTS_DIR = Path(__file__).parent / "reports"
JUDGE_SCORES = {"FAITHFUL": 1.0, "PARTIAL": 0.5, "UNFAITHFUL": 0.0}
JUDGE_PROMPT_VERSION = 2  # grounded in the retrieved contexts (v1 graded vs reference_answer)
LF = "\n"


# ------------------------------------------------------------------------------ setup


def _with_db(url: str, db: int) -> str:
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path=f"/{db}"))


def _db_of(url: str) -> int:
    path = urlsplit(url).path.strip("/")
    return int(path) if path else 0


async def prepare_eval_redis(settings: Settings, db: int) -> str:
    """Flush the eval database and copy each tenant's index_version into it."""
    from redis.asyncio import Redis

    from app.cache.keys import index_version_key

    if _db_of(settings.redis_url) == db:
        raise SystemExit(f"--redis-db {db} is the main database; pick another one")
    eval_url = _with_db(settings.redis_url, db)
    source, target = Redis.from_url(settings.redis_url), Redis.from_url(eval_url)
    try:
        missing = []
        await target.flushdb()
        for tenant in sorted(settings.known_tenants):
            version = await source.get(index_version_key(tenant))
            if version is None:
                missing.append(tenant)
            else:
                await target.set(index_version_key(tenant), version)
        if missing:
            raise SystemExit(
                f"no index_version for {missing}: ingest first "
                "(python -m app.rag.ingest data/mock_docs)"
            )
    finally:
        await source.aclose()
        await target.aclose()
    return eval_url


async def build_offline_pipeline(settings: Settings, audit_path: Path):  # noqa: ANN201
    """The real RagPipeline over in-memory Qdrant + fakes. Quality numbers are meaningless
    here; the harness, the ACL leak gates and the cache groups are still exercised."""
    from fakeredis import FakeAsyncRedis
    from qdrant_client import AsyncQdrantClient

    from app.cache.layered import LayeredCache
    from app.gateway.providers.fake import FakeLLM
    from app.gateway.providers.template import TemplateProvider
    from app.gateway.router import load_models_config
    from app.rag.embeddings import FakeEmbedder
    from app.rag.ingest import load_corpus, upsert
    from app.rag.pipeline import RagPipeline
    from app.rag.reranker import FakeReranker
    from app.rag.retriever import HybridRetriever
    from app.security.audit import AuditLogger

    qdrant = AsyncQdrantClient(location=":memory:")
    redis = FakeAsyncRedis()
    await upsert(
        load_corpus(CORPUS_ROOT, settings.known_tenants),
        client=qdrant,
        embedder=(embedder := FakeEmbedder()),
        redis=redis,
        collection=settings.qdrant_collection,
    )
    config = load_models_config(settings.models_config_path)
    llm = FakeLLM("openrouter", default="The documents cover this [1].")
    pipeline = RagPipeline(
        retriever=HybridRetriever(qdrant, embedder, settings.qdrant_collection),
        router=ModelRouter(
            config=config, providers={"openrouter": llm, "local": TemplateProvider()}
        ),
        prompt=load_prompt(settings.prompts_dir, "rag_qa", settings.answer_prompt_version),
        retrieve_top_k=settings.retrieve_top_k,
        context_top_n=settings.rerank_top_n,
        reranker=FakeReranker(),
        rerank_threshold=settings.rerank_threshold,
        audit=AuditLogger(audit_path),
        cache=LayeredCache(redis, l1_size=100, l1_ttl_s=60, l2_ttl_s=60),
    )
    return pipeline, qdrant


# ------------------------------------------------------------------------------ running


def _audit_by_trace(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            records[rec.get("trace_id", "")] = rec
    return records


class SpanCapture(logging.Handler):
    """Collects the rerank span's `context_refs` per trace_id (the span is logged inside the
    request's context, so the trace_id is the live one). Attached to the root logger."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.context_refs: dict[str, list[str]] = {}

    def emit(self, record: logging.LogRecord) -> None:
        fields = getattr(record, _FIELDS_ATTR, None) or {}
        if record.getMessage() == "span" and fields.get("span") == "rerank":
            self.context_refs[current_trace_id() or ""] = list(fields.get("context_refs") or [])


async def fetch_contexts(qdrant: Any, collection: str, refs: list[str]) -> list[RetrievedChunk]:
    """Read chunks back by `doc_id#chunk_index`, in the given order. Eval-side only: the refs
    come from the gateway's own span for this caller, so nothing outside its ACL is read."""
    from qdrant_client import models

    chunks = []
    for ref in refs:
        doc_id, _, index = ref.rpartition("#")
        points, _ = await qdrant.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                    models.FieldCondition(
                        key="chunk_index", match=models.MatchValue(value=int(index))
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
        )
        if points:
            payload = points[0].payload or {}
            chunks.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    chunk_index=int(index),
                    text=payload.get("text", ""),
                    title=payload.get("title"),
                )
            )
    return chunks


async def run_case(
    client: httpx.AsyncClient,
    case: GoldenCase,
    settings: Settings,
    audit_path: Path,
    capture: SpanCapture,
) -> CaseOutcome:
    token = issue_token(
        tenant_id=case.tenant_id,
        role=case.role,
        user_id=f"eval_{case.id.replace('-', '_').lower()}",
        settings=settings,
        ttl=timedelta(minutes=10),
    )
    started = time.perf_counter()
    resp = await client.post(
        "/v1/chat",
        json={"query": case.question},
        headers={"Authorization": f"Bearer {token}"},
    )
    latency_ms = (time.perf_counter() - started) * 1000
    is_json = resp.headers.get("content-type", "").startswith("application/json")
    body = resp.json() if is_json else None
    trace_id = (body or {}).get("trace_id") or resp.headers.get("X-Trace-ID", "")
    audit = _audit_by_trace(audit_path).get(trace_id)
    return CaseOutcome(
        status_code=resp.status_code,
        latency_ms=latency_ms,
        response=body if resp.status_code == 200 else None,
        retrieved_doc_ids=audit["retrieved_doc_ids"] if audit else None,
        decision=audit["decision"] if audit else None,
        context_refs=capture.context_refs.get(trace_id),
    )


_VERDICT_LINE = re.compile(r"^\W*(FAITHFUL|PARTIAL|UNFAITHFUL)\b", re.IGNORECASE)


def parse_verdict(text: str) -> str | None:
    """The judge's final verdict: the LAST line that starts with a verdict word. Judges
    sometimes write a verdict, reconsider, and end with a corrected one (seen live on D-08:
    "UNFAITHFUL: ... Checking more carefully ... FAITHFUL: ..."), so the first word is wrong."""
    for line in reversed(text.splitlines()):
        m = _VERDICT_LINE.match(line)
        if m:
            return m.group(1).upper()
    return None


async def judge_answers(
    router: ModelRouter,
    settings: Settings,
    pairs: list[tuple[GoldenCase, CaseOutcome]],
    qdrant: Any,
) -> tuple[list[CaseOutcome], float]:
    """Grounded LLM-as-judge through the gateway router (route `answer`): each answered case
    is graded against the exact contexts its answer was generated from."""
    from app.cost.pricing import compute_cost_usd

    prompt = load_prompt(settings.prompts_dir, "judge", JUDGE_PROMPT_VERSION)
    out, cost = [], 0.0
    for case, outcome in pairs:
        resp = outcome.response
        if not resp or resp.get("refused") or not outcome.context_refs:
            out.append(outcome)
            continue
        contexts = await fetch_contexts(qdrant, settings.qdrant_collection, outcome.context_refs)
        if not contexts:
            out.append(outcome)
            continue
        wrapped = wrap_documents(contexts)  # retrieved text is untrusted, for the judge too
        messages = [
            ChatMessage(role="system", content=prompt.render_system(doc_tag=wrapped.tag)),
            ChatMessage(
                role="user",
                content=prompt.render_user(
                    documents=wrapped.text, question=case.question, answer=resp["answer"]
                ),
            ),
        ]
        try:
            routed = await router.chat("answer", messages)
        except Exception as exc:  # noqa: BLE001 - a failed judgement is just unscored
            print(f"  judge {case.id}: failed ({type(exc).__name__})", flush=True)
            out.append(outcome)
            continue
        cost += compute_cost_usd(routed.response.usage, router.spec(routed.model_key)) or 0.0
        text = routed.response.text.strip()
        verdict = parse_verdict(text)
        score = JUDGE_SCORES.get(verdict or "")
        line = next((ln for ln in reversed(text.splitlines()) if verdict and verdict in ln), text)
        print(f"  judge {case.id:<5} {verdict or '?':<11} {line.strip()[:100]}", flush=True)
        out.append(replace(outcome, judge_score=score, judge_verdict=text[:600]))
    return out, cost


async def run_all(
    cases: list[GoldenCase], settings: Settings, *, offline: bool, judge: bool, redis_db: int
) -> tuple[list[CaseOutcome], float | None]:
    from app.gateway.factory import build_providers
    from app.gateway.router import load_models_config
    from app.main import app, get_pipeline

    with tempfile.TemporaryDirectory(prefix="rag-eval-") as tmp:
        audit_path = Path(tmp) / "audit.jsonl"
        overrides: dict[str, Any] = {"audit_log_path": audit_path}
        if offline:
            overrides |= {"embedder": "fake", "reranker": "fake"}
        else:
            overrides["redis_url"] = await prepare_eval_redis(settings, redis_db)
        eval_settings = settings.model_copy(update=overrides)

        app.dependency_overrides[get_settings] = lambda: eval_settings
        offline_qdrant = None
        if offline:
            pipeline, offline_qdrant = await build_offline_pipeline(eval_settings, audit_path)
            app.dependency_overrides[get_pipeline] = lambda: pipeline
        outcomes: list[CaseOutcome] = []
        judge_cost: float | None = None
        capture = SpanCapture()
        root = logging.getLogger()
        try:
            async with app.router.lifespan_context(app):
                # Keep the report readable (console handlers WARNING+) while INFO span events
                # still reach the capture handler.
                for handler in root.handlers:
                    handler.setLevel(logging.WARNING)
                root.setLevel(logging.INFO)
                root.addHandler(capture)
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://eval", timeout=300.0
                ) as client:
                    for i, case in enumerate(cases, 1):
                        outcome = await run_case(client, case, eval_settings, audit_path, capture)
                        outcomes.append(outcome)
                        resp = outcome.response or {}
                        print(
                            f"  [{i:>2}/{len(cases)}] {case.id:<5} {case.category.value:<13} "
                            f"http={outcome.status_code} {outcome.decision or '-':<18} "
                            f"{outcome.latency_ms / 1000:6.1f}s "
                            f"model={resp.get('model_used') or '-'}",
                            flush=True,
                        )
                if judge:
                    print("\nLLM-as-judge (grounded faithfulness vs the retrieved contexts):")
                    cfg = load_models_config(eval_settings.models_config_path)
                    if offline:
                        from app.gateway.providers.fake import FakeLLM
                        from app.gateway.providers.template import TemplateProvider

                        providers: dict[str, Any] = {
                            "openrouter": FakeLLM("openrouter", default="FAITHFUL: offline"),
                            "local": TemplateProvider(),
                        }
                    else:
                        providers = dict(build_providers(cfg, eval_settings, app.state.http))
                    outcomes, judge_cost = await judge_answers(
                        ModelRouter(config=cfg, providers=providers),
                        eval_settings,
                        list(zip(cases, outcomes, strict=True)),
                        offline_qdrant if offline else app.state.qdrant,
                    )
        finally:
            root.removeHandler(capture)
            app.dependency_overrides.clear()
            if offline_qdrant is not None:
                await offline_qdrant.close()
    return outcomes, judge_cost


# ------------------------------------------------------------------------------ reporting


def _config_snapshot(settings: Settings, offline: bool) -> dict[str, Any]:
    return {
        "mode": "offline" if offline else "live",
        "embedder": "fake" if offline else settings.embedder,
        "reranker": "fake" if offline else settings.reranker,
        "rerank_model_name": settings.rerank_model_name,
        "rerank_threshold": settings.rerank_threshold,
        "retrieve_top_k": settings.retrieve_top_k,
        "rerank_top_n": settings.rerank_top_n,
        "answer_prompt_version": settings.answer_prompt_version,
        "rewrite_prompt_version": settings.rewrite_prompt_version,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
    }


def _case_line(r: CaseResult) -> str:
    flags = []
    if r.unauthorized_leaks:
        flags.append(f"LEAK{r.unauthorized_leaks}")
    if r.cross_tenant_leaks:
        flags.append(f"XTENANT{r.cross_tenant_leaks}")
    if r.injection_hits:
        flags.append(f"INJECTED{r.injection_hits}")
    if r.isolation_hits:
        flags.append(f"ISOLATION{r.isolation_hits}")
    if r.error:
        flags.append(r.error)
    if not r.refusal_correct:
        flags.append("missed-refusal" if r.expect_refusal else "false-refusal")
    elif r.answer_success is False:
        flags.append(f"keywords {r.keyword_hits}/{r.keyword_total} missed {r.keywords_missed}")
    if r.citation_correct is False:
        flags.append("citation")
    if r.cache_expectation_met is False:
        flags.append("cache")
    if r.truncated:
        flags.append("TRUNCATED")
    status = "ok " if not flags else "-- "
    return f"  {status}{r.id:<5} {r.category:<13} " + ("; ".join(flags) if flags else "")


def _baseline_payload(report: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "id", "category", "refused", "refusal_correct", "answer_success", "keyword_hits",
        "keyword_total", "decision", "cited_doc_ids", "citation_correct", "latency_ms",
        "cost_usd", "model_used", "judge_score", "keywords_missed",
    )  # fmt: skip
    return {
        "generated_at": report["generated_at"],
        "config": report["config"],
        "summary": report["summary"],
        "cases": [{k: c[k] for k in keep} for c in report["cases"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="validate the golden set only")
    parser.add_argument("--offline", action="store_true", help="fakes only, no infra needed")
    parser.add_argument("--judge", action="store_true", help="add LLM-as-judge faithfulness")
    parser.add_argument("--compare", type=Path, default=BASELINE_PATH, help="baseline JSON")
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--redis-db", type=int, default=15, help="Redis DB used by the eval")
    parser.add_argument("--only", nargs="*", help="run only these case ids")
    args = parser.parse_args()

    try:
        cases = load_golden()
    except (ValidationError, ValueError) as exc:
        print(f"golden set invalid: {exc}")
        return 2

    print(f"{len(cases)} cases")
    for k, v in category_mix(cases).items():
        print(f"  {k:<28} {v:6.1%}  (target {TARGET_MIX[k]:.0%})")
    if args.dry_run:
        return 0
    if args.only:
        cases = [c for c in cases if c.id in set(args.only)]

    settings = get_settings()
    config = _config_snapshot(settings, args.offline)
    print("\nconfig: " + ", ".join(f"{k}={v}" for k, v in config.items()))
    if args.offline:
        print("OFFLINE: fake embedder / reranker / LLM; only the harness and gates are real.")
    print()

    started = time.perf_counter()
    outcomes, judge_cost = asyncio.run(
        run_all(cases, settings, offline=args.offline, judge=args.judge, redis_db=args.redis_db)
    )
    docs = load_corpus_docs()
    results = [score_case(c, o, docs) for c, o in zip(cases, outcomes, strict=True)]
    summary = summarize(results)
    if judge_cost is not None:
        summary["judge_cost_usd"] = round(judge_cost, 6)

    print("\nPer case:")
    for r in results:
        print(_case_line(r))

    baseline = None
    if not args.offline and args.compare and args.compare.exists():
        baseline = json.loads(args.compare.read_text(encoding="utf-8"))
    base_summary = baseline["summary"] if baseline else None
    label = f"{args.compare} ({baseline['generated_at'][:19]})" if baseline else "none"
    print(f"\nComparison with baseline: {label}")
    rows = compare(summary, base_summary)
    print(render_table(rows))

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": config,
        "baseline": str(args.compare) if baseline else None,
        "duration_s": round(time.perf_counter() - started, 1),
        "summary": summary,
        "cases": result_dicts(results),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = REPORTS_DIR / f"eval_{'offline_' if args.offline else ''}{stamp}.json"
    path.write_text(json.dumps(report, indent=2) + LF, encoding="utf-8", newline=LF)
    print(f"\nreport: {path.relative_to(ROOT)}  ({report['duration_s']} s)")
    if args.save_baseline:
        if args.offline:
            print("refusing to save an offline run as the baseline")
        else:
            BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps(_baseline_payload(report), indent=2) + LF
            BASELINE_PATH.write_text(text, encoding="utf-8", newline=LF)
            print(f"baseline saved: {BASELINE_PATH.relative_to(ROOT)}")

    failures = hard_failures(summary)
    regressions = [r.metric for r in rows if r.flag == "REGRESSION"]
    print(
        f"\nunauthorized leaks = {summary['unauthorized_leaks']}, "
        f"cross-tenant leaks = {summary['cross_tenant_leaks']}, "
        f"successful injections = {summary['injection_successes']}, "
        f"cache isolation violations = {summary['isolation_violations']}"
    )
    if regressions:
        print(f"REGRESSION (soft): {', '.join(regressions)}")
    if failures:
        print(f"FAIL: {', '.join(failures)}")
        return 1
    print("PASS: all hard gates hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
