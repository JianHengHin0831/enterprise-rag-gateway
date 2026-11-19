"""Measure query-rewrite latency per candidate model through OpenRouter.

Every call goes through `ModelRouter` (route `rewrite`, the real rewrite prompt and
`QueryRewriter.build_messages`), one attempt per call (no retry, no fallback), so the
numbers are single-request latency. Calls cost real money (a few hundred tokens each);
`--dry-run` builds everything and prints the plan without calling a model.

python -m scripts.bench_rewrite [--runs 5] [--timeout 15] [--models ID ...] [--dry-run]
"""

import argparse
import asyncio
import copy
import statistics
import time
from dataclasses import dataclass, field

import httpx

from app.config import get_settings
from app.gateway.base import LLMError
from app.gateway.factory import build_providers
from app.gateway.router import ModelRouter, load_models_config
from app.rag.prompts import load_prompt
from app.rag.rewrite import QueryRewriter
from app.schemas import ChatRequest, ChatTurn

DEFAULT_MODELS = [
    "qwen/qwen3.8-flash",  # current qwen-turbo mapping
    "qwen/qwen3.7-flash",
    "anthropic/claude-haiku-5.5",
]

# Synthetic requests of the two kinds that get rewritten: multi-turn and short queries.
SAMPLES = [
    ChatRequest(
        query="and for interns?",
        history=[
            ChatTurn(role="user", content="How many paid vacation days do new employees get?"),
            ChatTurn(role="assistant", content="New full-time employees get 15 days [1]."),
        ],
    ),
    ChatRequest(
        query="how often must I change it?",
        history=[
            ChatTurn(role="user", content="What is the password policy for laptops?"),
            ChatTurn(role="assistant", content="Passwords need at least 14 characters [1]."),
        ],
    ),
    ChatRequest(query="VPN setup"),
    ChatRequest(query="travel expense limit"),
]

BENCH_KEY = "bench"


@dataclass
class Result:
    model: str
    latencies_ms: list[float] = field(default_factory=list)
    errors: dict[str, int] = field(default_factory=dict)
    samples: list[str] = field(default_factory=list)


def _router_for(base: dict, model_id: str, timeout_s: float, providers: dict) -> ModelRouter:
    cfg = copy.deepcopy(base)
    template = cfg["models"]["qwen-turbo"]
    cfg["models"] = {
        BENCH_KEY: {**template, "provider": "openrouter", "model": model_id, "timeout_s": timeout_s}
    }
    cfg["routes"] = {"rewrite": [BENCH_KEY]}
    cfg["retry"] = {**cfg["retry"], "max_attempts": 1, "request_deadline_s": timeout_s}
    return ModelRouter(config=cfg, providers=providers)


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="OpenRouter ids")
    parser.add_argument("--runs", type=int, default=5, help="calls per model")
    parser.add_argument("--timeout", type=float, default=15.0, help="per-call timeout (s)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, call nothing")
    args = parser.parse_args()

    settings = get_settings()
    base = load_models_config(settings.models_config_path)
    prompt = load_prompt(settings.prompts_dir, "rewrite", settings.rewrite_prompt_version)

    async with httpx.AsyncClient() as http:
        providers = build_providers(base, settings, http)
        if "openrouter" not in providers:
            raise SystemExit("RAG_OPENROUTER_API_KEY is not set; nothing to benchmark.")

        routers = {m: _router_for(base, m, args.timeout, providers) for m in args.models}
        # build_messages does not depend on the router; any one will do.
        rewriter = QueryRewriter(
            router=next(iter(routers.values())),
            prompt=prompt,
            timeout_s=args.timeout,
            min_words=settings.rewrite_min_words,
        )
        messages = [rewriter.build_messages(s) for s in SAMPLES]
        total = len(args.models) * args.runs
        print(
            f"{len(args.models)} models x {args.runs} runs = {total} calls"
            f" (timeout {args.timeout}s)"
        )
        if args.dry_run:
            for m in args.models:
                print(f"  would call {m}")
            return

        results = {m: Result(m) for m in args.models}
        # Round-robin across models so a slow period does not hit one model only.
        for i in range(args.runs):
            msgs = messages[i % len(messages)]
            for model_id, router in routers.items():
                res = results[model_id]
                t0 = time.perf_counter()
                try:
                    routed = await router.chat("rewrite", msgs)
                except (LLMError, TimeoutError) as exc:
                    name = type(exc).__name__
                    res.errors[name] = res.errors.get(name, 0) + 1
                    print(f"  {model_id:<32} run {i + 1}: {name}")
                    continue
                ms = (time.perf_counter() - t0) * 1000
                res.latencies_ms.append(ms)
                text = (routed.response.text.strip().splitlines() or [""])[0][:70]
                if len(res.samples) < len(SAMPLES):
                    res.samples.append(text)
                print(f"  {model_id:<32} run {i + 1}: {ms:7.0f} ms  {text!r}")

    limit_ms = settings.rewrite_timeout_s * 1000
    print(f"\nrewrite timeout in settings: {settings.rewrite_timeout_s}s")
    print(
        f"{'model':<32} {'ok':>3} {'p50':>7} {'p90':>7} {'max':>7} {'>3s':>4} {'>limit':>6}  errors"
    )
    for res in results.values():
        lat = res.latencies_ms
        if lat:
            p50, p90, mx = statistics.median(lat), _pct(lat, 0.9), max(lat)
            over3 = sum(v > 3000 for v in lat)
            over_limit = sum(v > limit_ms for v in lat)
            row = f"{p50:7.0f} {p90:7.0f} {mx:7.0f} {over3:>4} {over_limit:>6}"
        else:
            row = f"{'-':>7} {'-':>7} {'-':>7} {'-':>4} {'-':>6}"
        errors = ", ".join(f"{k}={v}" for k, v in res.errors.items()) or "-"
        print(f"{res.model:<32} {len(lat):>3} {row}  {errors}")


if __name__ == "__main__":
    asyncio.run(main())
