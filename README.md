# Enterprise RAG Gateway

An internal knowledge-base QA gateway: **multi-tenancy + role-based isolation + LLM gateway resilience + anti-hallucination loop + cost control + end-to-end observability**.

It simulates the HR / engineering / company-wide document libraries of a fictional tech company, with two tenants (`acme_corp`, `beta_inc`) and four roles (`intern`, `employee`, `eng`, `hr`).

> Status: M0 skeleton done, M1 in progress. See [ROADMAP.md](ROADMAP.md) / [STATUS.md](STATUS.md).

---

## Core features

| Capability | Implementation |
|---|---|
| Identity & access | `tenant_id / role` come from a verified JWT; Qdrant payload pre-filtering **at query level** (filter on every prefetch), never post-filtering |
| Injection defense | Retrieved text wrapped in random-nonce tags with forged tags neutralized; injection patterns screened at ingest; system prompt declares documents non-executable |
| Anti-hallucination | Hybrid retrieval (dense + sparse, RRF) → rerank → low-score refusal gate (no LLM call) → mandatory `[n]` citations → citation range check |
| Multi-model resilience | Provider adapter abstraction; retryable vs non-retryable error taxonomy; three-state circuit breaker per `provider:model`; exponential backoff + jitter + overall deadline; fallback chain |
| Streaming | SSE; retry/fallback **only before the first token**, after that a failure ends the stream with `event: error` |
| Cost | tiktoken estimate + actual usage reconciliation; Redis daily budget with "reserve → settle" to prevent concurrent overspend |
| Cache | L1 in-process TTLCache + L2 Redis; key binds tenant / role / prompt version / rewrite prompt version / index version; degraded answers are never cached |
| Observability | `contextvars` trace_id; `Span` timing for rewrite / retrieve / rerank / llm; single-line JSON logs with automatic PII masking; `rag_request_summary` event (tokens estimated vs actual, cost, per-step ms); append-only audit log (query hash only, every decision incl. client disconnects); optional LangFuse export over OTLP |
| Evaluation | Golden set (answerable / no-answer traps / unauthorized / cross-tenant / injection), compared against a baseline |

---

## Request flow

```
Client ──(Authorization: Bearer <JWT>)──►
 1. Auth: verify signature → Identity; create trace_id
 2. Budget: reserve estimated tokens on budget:{tenant}:{date} (exceeded → 429 / downgrade)
 3. Input safety: injection pattern screening
 4. Query rewrite (small model, only multi-turn / short queries; falls back to the raw query)
 5. Cache lookup: key = sha256(tenant, role, prompt_v, rewrite_v, index_v, query)
      └─ hit → stream back (cache_hit=true, cost=0)
 6. Hybrid retrieval: Qdrant prefetch[dense, sparse] + RRF, ACL filter on every prefetch → Top-20
 7. Rerank → Top-5; Top-1 < threshold → fixed refusal (no LLM call)
 8. ModelRouter: circuit breaker + backoff + fallback chain → SSE stream (with [n] citations)
 9. Post: citation check → should_cache → write cache → settle budget → audit log + summary event
```

---

## Quick start

```bash
cp .env.example .env            # fill in API keys
docker compose up -d qdrant redis
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"     # Linux/macOS: .venv/bin/pip
.venv/Scripts/pytest tests/ -v
.venv/Scripts/uvicorn app.main:app --reload --port 8000
```

Open the demo console at <http://localhost:8000/> (served from `app/web/`). Pick a tenant and
role, sign in, and ask questions; switch personas to show access control. Persona tokens come from
the dev-only `POST /dev/token` route, which returns 404 unless `RAG_ENV=dev` - set `RAG_ENV=prod`
anywhere the demo is exposed publicly. You can also paste a token minted by the script below.

Mint a test identity and call the API:

```bash
TOKEN=$(.venv/Scripts/python -m scripts.mint_token --tenant acme_corp --role hr --user u_hr_01)
curl -s localhost:8000/v1/chat -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" -d '{"query": "How many performance rating tiers are there?"}'
```

Start the whole stack: `docker compose up --build`

---

## Layout

```
app/
  main.py            FastAPI entry: /health /ready /v1/chat, trace middleware
  config.py          Env-var settings (RAG_ prefix)
  schemas.py         Identity, request/response, retrieval models
  security/          auth (JWT) · acl (Qdrant filter) · injection (wrapping/screening) · audit
  observability/     trace (contextvars + Span) · sanitizer (masking) · logging (JSON)
  gateway/           base (provider interface / error taxonomy) · circuit_breaker · router · providers/
  cost/              tokens (tiktoken estimate) · pricing · budget (reserve/settle)
  cache/             keys (isolated keys + should_cache) · layered (L1/L2)
  rag/               ingest · embeddings · retriever · reranker · citations · prompts · pipeline
configs/             models.yaml (routing/pricing/breaker/retry) · prompts/ (versioned prompts)
data/mock_docs/      <tenant>/<access>/*.md, front matter declares tenant_id / allowed_roles
evals/               golden_set.json · run_eval.py · baselines/
scripts/             mint_token.py (issues dev JWTs)
tests/               unit tests + FakeLLM integration tests (future-milestone acceptance tests pre-written as skips)
```

---

## Refusal threshold calibration (M3)

The low-score gate refuses with a fixed text, without calling the LLM, when the reranker's top-1
score is below `RAG_RERANK_THRESHOLD`. The threshold is only meaningful for the reranker it was
calibrated on, so it is set by sweeping the golden set through the real path (verified JWT →
ACL-filtered hybrid retrieval Top-20 → `BAAI/bge-reranker-v2-m3` Top-5), with no LLM involved:

```bash
.venv/Scripts/python -m evals.sweep_threshold --svg docs/threshold_curve.svg   # ~15-30 min on CPU
.venv/Scripts/python -m evals.sweep_threshold --from-report evals/reports/threshold_sweep.json
```

![Threshold vs gate accuracy, answer recall, refusal recall and refusal rate](docs/threshold_curve.svg)

51 golden cases: 32 should be answered, 19 should be refused (`expect_refusal`). *Answer recall* =
should-answer cases that pass the gate; *refusal recall* = should-refuse cases the gate refuses on
its own; *gate accuracy* = correct gate decisions over all cases.

| Threshold | Answer recall | Refusal recall | Gate accuracy | Refusal rate (all) |
|---|---|---|---|---|
| 0.00 | 100.0% | 0.0% | 62.7% | 0.0% |
| 0.01 | 96.9% | 63.2% | 84.3% | 25.5% |
| 0.02 | 96.9% | 73.7% | 88.2% | 29.4% |
| **0.03 (chosen)** | **96.9%** | **78.9%** | **90.2%** | **31.4%** |
| 0.05 | 93.8% | 84.2% | 90.2% | 35.3% |
| 0.10 | 75.0% | 89.5% | 80.4% | 49.0% |
| 0.35 (old placeholder) | 50.0% | 94.7% | 66.7% | 66.7% |
| 0.50 | 46.9% | 94.7% | 64.7% | 68.6% |

**Chosen: 0.03.** Rule: highest gate accuracy among thresholds keeping answer recall ≥ 95%, middle
of the best plateau. A blocked answer cannot be recovered, while weak context that passes the gate is
still refused by the LLM (prompt rule), so the gate is tuned to rarely block.

What the sweep showed:
- Unauthorized, cross-tenant and most no-answer questions score ≤ 0.022 (ACL filtering leaves only
  unrelated chunks), while answerable ones mostly score ≥ 0.04. The gap sits around 0.03.
- Cross-encoder scores fall sharply for multi-part questions (one chunk rarely answers all parts):
  "What is the expense submission deadline and the domestic meal allowance?" scores 0.07-0.10
  although the right document ranks first. The old 0.35 placeholder would have refused half of
  the answerable questions.
- The gate cannot catch everything: N-05 (on-call rotation, 0.29) and the prompt-injection
  question E-06 (0.80) pass, and are left to the LLM's refusal rule and the injection defenses.
  The single answerable case below 0.03 is the false-premise P-01 ("deploy with Helm"), where a
  refusal is an acceptable outcome.
- After reranking, the expected document is in the Top-5 for 32/32 cases that name one.

---

## Design trade-offs (interview cheat sheet)

- **Why not trust an `X-User-Role` header?** Clients can forge it. Identity only comes from a verified JWT (issued by an upstream SSO/gateway in production); the ACL filter is derived only from that identity, and LLM rewrite output can never influence it.
- **Why filter at query level?** Taking Top-K first and filtering in Python pushes authorized results out of the Top-K, and unauthorized documents have already been loaded into memory.
- **Why is the cache lookup after query rewrite?** In a multi-turn chat "what is its limit?" depends on context; caching on the raw query would return conversation A's answer to conversation B.
- **Why are degraded answers not cached?** Otherwise a fallback answer produced during an outage would be served for an hour.
- **Why only fall back before the first streamed token?** Sent tokens cannot be recalled; switching models mid-answer stitches together two styles and possibly contradictory content.
- **Why reserve budget up front?** "Check, then spend" overshoots under concurrency.
- **Why is breaker state not shared across processes?** A demo trade-off: per-worker breakers still protect the downstream; sharing needs Redis and adds a new failure point.
- **How is the refusal threshold chosen?** Score distributions differ between rerankers, so the threshold is calibrated by sweeping the golden set (see [Refusal threshold calibration](#refusal-threshold-calibration-m3)). It favours answer recall: the LLM can still refuse weak context, but nothing recovers a wrongly blocked answer.
- **Why rerank on the full chunk instead of trusting RRF scores?** RRF scores are rank-based and not comparable across queries, so they cannot drive an absolute refusal threshold; a cross-encoder score can.

---

## Metrics (filled in at M4)

| Metric | Value |
|---|---|
| Answerable hit rate | – |
| No-answer refusal accuracy | – |
| Unauthorized leaks | – (target 0) |
| Successful injections | – (target 0) |
| P50 / P95 latency | – |
| Average cost per request | – |
