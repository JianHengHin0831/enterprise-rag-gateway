"""Load and concurrency measurements (no LLM calls, $0).

python -m scripts.loadtest                  # all scenarios
python -m scripts.loadtest --only cache     # one scenario: cache | hit | refusal | budget

Scenarios:
- cache    In-process micro-benchmark of `LayeredCache.get` against the real Redis (a
           dedicated database, flushed): L1 hit, L2 hit (Redis + JSON decode), miss.
- hit      HTTP against the running API: one question is answered once, then repeated at
           the given concurrency. Every request is an L1 hit, so this is the gateway's own
           ceiling: JWT verification, query screening, cache lookup, audit log, response.
- refusal  HTTP against the running API: unique out-of-scope questions, so every request runs
           embedding, ACL-filtered hybrid retrieval and the reranker, then is refused by the
           low-score gate without an LLM call. This is the retrieval stack's throughput.
- budget   Several OS processes reserve tokens concurrently against the real Redis: the naive
           check-then-spend pattern versus `BudgetGuard` (atomic reserve -> settle). Reports
           how far each one overshoots the daily limit.

The API must be running (`uvicorn app.main:app --port 8000`) for `hit` and `refusal`, with
the corpus ingested. Tokens are minted locally with `RAG_JWT_SECRET`, as `scripts.mint_token`
does. Answer-path throughput is not measured here: it is bounded by the LLM provider's rate
limits, and its latency is in the golden-set eval.
"""

import argparse
import asyncio
import logging
import multiprocessing as mp
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta

import httpx
from redis.asyncio import Redis

from app.cache.layered import LayeredCache
from app.config import get_settings
from app.cost.budget import BudgetGuard
from app.schemas import ChatResponse, Role
from app.security.auth import issue_token

BENCH_REDIS_DB = 14  # never the app's database; flushed by the scenarios that use it
HIT_QUESTION = "What is the guest Wi-Fi policy?"
REFUSAL_QUESTION = "What is the strike price of the company's stock options"


@dataclass
class Stats:
    name: str
    n: int
    errors: int
    wall_s: float
    latencies_ms: list[float]

    def row(self) -> str:
        lat = sorted(self.latencies_ms) or [0.0]

        def pct(p: float) -> float:
            return lat[min(len(lat) - 1, round(p * (len(lat) - 1)))]

        return (
            f"| {self.name} | {self.n} | {self.errors} | {self.n / self.wall_s:,.1f} "
            f"| {statistics.median(lat):,.3f} | {pct(0.95):,.3f} | {pct(0.99):,.3f} |"
        )


HEADER = (
    "| Scenario | Requests | Errors | Throughput (/s) | P50 ms | P95 ms | P99 ms |\n"
    "|---|---|---|---|---|---|---|"
)


def _redis_url(db: int) -> str:
    base = get_settings().redis_url.rsplit("/", 1)[0]
    return f"{base}/{db}"


# ------------------------------------------------------------------------- cache


async def bench_cache(n: int) -> list[Stats]:
    redis = Redis.from_url(_redis_url(BENCH_REDIS_DB))
    await redis.flushdb()
    value = ChatResponse(trace_id="t", answer="A realistic answer of a few sentences. [1] " * 8)
    out = []
    try:
        cache = LayeredCache(redis, l1_size=4096, l1_ttl_s=3600, l2_ttl_s=3600)
        for i in range(n):
            await cache.set(f"k{i}", value)

        async def timed(name: str, make_cache) -> Stats:  # noqa: ANN001
            lat = []
            t0 = time.perf_counter()
            for i in range(n):
                c = make_cache()
                s = time.perf_counter()
                await c.get(f"k{i}" if name != "miss" else f"absent{i}")
                lat.append((time.perf_counter() - s) * 1000)
            return Stats(f"cache {name}", n, 0, time.perf_counter() - t0, lat)

        out.append(await timed("L1 hit", lambda: cache))
        # A fresh LayeredCache has an empty L1, so each get goes to Redis.
        out.append(
            await timed("L2 hit", lambda: LayeredCache(redis, l1_size=1, l1_ttl_s=1, l2_ttl_s=60))
        )
        out.append(
            await timed("miss", lambda: LayeredCache(redis, l1_size=1, l1_ttl_s=1, l2_ttl_s=60))
        )
    finally:
        await redis.flushdb()
        await redis.aclose()
    return out


# ------------------------------------------------------------------------- HTTP


def _token(role: Role) -> str:
    return issue_token(
        tenant_id="acme_corp",
        role=role,
        user_id="u_loadtest",
        settings=get_settings(),
        ttl=timedelta(hours=1),
    )


async def http_scenario(
    url: str, name: str, questions: list[str], concurrency: int, token: str
) -> Stats:
    sem = asyncio.Semaphore(concurrency)
    lat: list[float] = []
    errors = 0
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(base_url=url, timeout=120) as client:

        async def one(q: str) -> None:
            nonlocal errors
            async with sem:
                s = time.perf_counter()
                try:
                    r = await client.post("/v1/chat", json={"query": q}, headers=headers)
                    ok = r.status_code == 200
                except httpx.HTTPError:
                    ok = False
                lat.append((time.perf_counter() - s) * 1000)
                errors += not ok

        t0 = time.perf_counter()
        await asyncio.gather(*(one(q) for q in questions))
        wall = time.perf_counter() - t0
    return Stats(f"{name} (concurrency {concurrency})", len(questions), errors, wall, lat)


async def bench_hit(url: str, n: int, concurrency: int) -> Stats:
    token = _token(Role.INTERN)
    async with httpx.AsyncClient(base_url=url, timeout=120) as client:  # warm the cache
        r = await client.post(
            "/v1/chat",
            json={"query": HIT_QUESTION},
            headers={"Authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
    stats = await http_scenario(url, "HTTP cache hit", [HIT_QUESTION] * n, concurrency, token)
    return stats


async def bench_refusal(url: str, n: int, concurrency: int) -> Stats:
    # A run id keeps questions unique across runs, so none is served from the cache.
    run = uuid.uuid4().hex[:6]
    questions = [f"{REFUSAL_QUESTION} (ref {run}-{i})?" for i in range(n)]
    token = _token(Role.EMPLOYEE)
    # Warm-up: the first request loads the embedding and rerank models.
    await http_scenario(url, "warm-up", [f"{REFUSAL_QUESTION} (warm {run})?"], 1, token)
    return await http_scenario(url, "HTTP gate refusal", questions, concurrency, token)


# ------------------------------------------------------------------------- budget race


LIMIT, ESTIMATE = 10_000, 100  # the limit admits exactly 100 requests
LLM_CALL_S = 0.05  # simulated model latency between the budget check and the spend


async def _naive_worker(url: str, key: str, n: int) -> int:
    redis = Redis.from_url(url)

    async def one() -> bool:
        spent = int(await redis.get(key) or 0)  # check ...
        if spent + ESTIMATE > LIMIT:
            return False
        await asyncio.sleep(LLM_CALL_S)  # ... the LLM call ...
        await redis.incrby(key, ESTIMATE)  # ... then spend
        return True

    served = sum(await asyncio.gather(*(one() for _ in range(n))))
    await redis.aclose()
    return served


async def _guard_worker(url: str, tenant: str, n: int) -> int:
    redis = Redis.from_url(url)
    guard = BudgetGuard(redis, {tenant: LIMIT})

    async def one() -> bool:
        r = await guard.reserve(tenant, ESTIMATE)  # atomic reserve before the call
        if not r.accepted:
            return False
        await asyncio.sleep(LLM_CALL_S)
        await guard.settle(r, ESTIMATE)
        return True

    served = sum(await asyncio.gather(*(one() for _ in range(n))))
    await redis.aclose()
    return served


def _proc(kind: str, url: str, target: str, n: int, barrier, results) -> None:  # noqa: ANN001
    logging.getLogger("rag.budget").setLevel(logging.ERROR)  # expected rejections
    worker = _naive_worker if kind == "naive" else _guard_worker
    barrier.wait()  # every process has imported and connected: start together
    results.put(asyncio.run(worker(url, target, n)))


async def _flush(url: str) -> None:
    r = Redis.from_url(url)
    await r.flushdb()
    await r.aclose()


async def _read_spent(url: str, kind: str, target: str) -> int:
    r = Redis.from_url(url)
    if kind == "naive":
        value = int(await r.get(target) or 0)
    else:
        value = await BudgetGuard(r, {target: LIMIT}).spent(target)
    await r.aclose()
    return value


def budget_race(processes: int, per_process: int) -> list[str]:
    url = _redis_url(BENCH_REDIS_DB)
    rows = []
    for kind, target, label in (
        ("naive", "budget_race:naive", "check-then-spend (naive)"),
        ("guard", "race_guard", "BudgetGuard reserve -> settle"),
    ):
        asyncio.run(_flush(url))
        barrier, results = mp.Barrier(processes), mp.Queue()
        procs = [
            mp.Process(target=_proc, args=(kind, url, target, per_process, barrier, results))
            for _ in range(processes)
        ]
        for proc in procs:
            proc.start()
        served = sum(results.get() for _ in procs)
        for proc in procs:
            proc.join()
        spent = asyncio.run(_read_spent(url, kind, target))
        over = max(0, spent - LIMIT)
        rows.append(
            f"| {label} | {processes} x {per_process} | {LIMIT:,} ({LIMIT // ESTIMATE} req) "
            f"| {served} | {spent:,} | {over:,} ({over / LIMIT:.0%}) |"
        )
    asyncio.run(_flush(url))
    return rows


# ------------------------------------------------------------------------- main


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--only", choices=["cache", "hit", "refusal", "budget"])
    p.add_argument("--hit-requests", type=int, default=2000)
    p.add_argument("--refusal-requests", type=int, default=100)
    p.add_argument("--concurrency", type=int, default=20)
    args = p.parse_args()
    run = lambda s: args.only in (None, s)  # noqa: E731

    rows: list[str] = []
    if run("cache"):
        rows += [s.row() for s in asyncio.run(bench_cache(2000))]
    if run("hit"):
        rows.append(asyncio.run(bench_hit(args.url, args.hit_requests, args.concurrency)).row())
    if run("refusal"):
        stats = asyncio.run(bench_refusal(args.url, args.refusal_requests, args.concurrency))
        rows.append(stats.row())
    if rows:
        print(HEADER)
        print("\n".join(rows))
    if run("budget"):
        print()
        print("| Budget pattern | Processes x requests | Daily limit | Served | Spent | Over |")
        print("|---|---|---|---|---|---|")
        print("\n".join(budget_race(processes=8, per_process=50)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
