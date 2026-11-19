# Roadmap

After each milestone: tests green → tick the boxes here and update `STATUS.md` → `git commit`.
Unfinished code is marked `TODO(Mx)`; acceptance tests for future milestones are pre-written with `pytest.mark.skip(reason="Mx: ...")`.

---

## M0 · Project skeleton ✅

- [x] Directory layout, `pyproject.toml`, Docker Compose (API + Qdrant + Redis), Dockerfile with HEALTHCHECK
- [x] JWT verification → `Identity` (`tenant_id / role / user_id`)
- [x] ACL filter builder (tenant + role, at database query level)
- [x] PII / secret sanitizer + single-line JSON logs
- [x] `contextvars` trace_id + `Span` async context manager
- [x] Three-state circuit breaker (Closed / Open / Half-Open, injectable clock)
- [x] Isolated cache key + `should_cache` rules
- [x] Citation range check
- [x] Random-nonce wrapping of retrieved text + tag-escape defense + injection screening
- [x] Model pricing calculation, versioned prompt loader
- [x] Sample mock docs (2 tenants, 3 access levels, malicious injection docs, fictional PII)
- [x] Sample golden set with schema validation

---

## M1 · Core path + access control + logging (days 1–3) ✅

Goal: ingest docs with `tenant_id / allowed_roles`; `/v1/chat` runs the minimal "retrieve → answer" path (non-streaming, single model for now).

- [x] Grow mock docs to 15–20 (acme_corp + beta_inc, all / eng / hr, including 2 malicious injection docs)
- [x] `app/rag/embeddings.py`: integrate bge-m3 (dense + sparse; local FlagEmbedding or an API)
- [x] `app/rag/ingest.py`: chunk → embed → upsert, payload includes `tenant_id / allowed_roles / doc_id / chunk_index / index_version`; mark `suspicious` with `screen_text` at ingest
- [x] Qdrant collection: named vectors `dense` + `sparse`, payload indexes on `tenant_id` and `allowed_roles`
- [x] `app/rag/retriever.py`: Query API `prefetch(dense, sparse)` + RRF fusion, **ACL filter on every prefetch**
- [x] Non-streaming `chat()` for one OpenAI-compatible provider (via the router, never direct)
- [x] `/v1/chat` wires up: auth → trace → retrieve → wrap docs → generate → return `answer + citations + trace_id`
- [x] `/ready` checks Qdrant + Redis

**Acceptance**
- intern and hr ask the same salary question: different trace_ids in the logs, and the intern's `retrieved_doc_ids` contain no hr documents
- A beta_inc user never retrieves an acme_corp chunk (integration test asserting on retrieval results)

---

## M2 · Multi-model gateway + breaker/fallback + budget + cache + SSE (days 4–6) ✅

- [x] Provider adapters: OpenAI-compatible + Anthropic (separate `system` parameter, different usage field names)
- [x] Error taxonomy: retryable (timeout / 429 / 5xx) vs non-retryable (400 / 401 / 403 / content refusal); non-retryable errors do not count toward the breaker
- [x] `ModelRouter`: fallback chain from `configs/models.yaml`; exponential backoff + full jitter; honour `Retry-After`; overall deadline
- [x] Breakers isolated per `provider:model`; response metadata includes `model_used`, `degraded`
- [x] **SSE streaming**: retry/fallback before the first token; after it, failures send `event: error` and end
- [x] `BudgetGuard`: Redis `budget:{tenant}:{YYYY-MM-DD}` reserve → settle with actual usage; over budget → 429 or downgrade to the small model
- [x] L1 `TTLCache(500, 60s)` + L2 Redis (1h); `index_version` stored in Redis, cached briefly in L1
- [x] Query rewrite (small model): only for multi-turn or short queries; on failure/timeout fall back to the raw query

**Acceptance**
- FakeLLM primary returns 429 five times in a row → the 6th call skips the primary and falls back, `degraded=true` (`tests/test_router_fallback.py`)
- Asking the same question twice: the second answer has `cache_hit=true`, `cost_usd=0`
- 20 concurrent requests at the budget limit: actual spend never exceeds the daily budget + one request's maximum

---

## M3 · Anti-hallucination + security audit (days 7–9)

- [ ] Rerank (external API or local bge-reranker), Top-20 → Top-5
- [ ] Low-score refusal gate: Top-1 score < threshold → fixed refusal, no LLM call
- [ ] **Calibrate the threshold by sweeping the golden set** (threshold vs accuracy/refusal-rate curve), add it to the README
- [ ] After streaming ends, run the citation check on the full text; on `citation_mismatch=true` append a disclaimer and record the metric
- [ ] Audit log: `who / tenant / role / query_hash / retrieved_doc_ids / decision / ts` (append-only JSONL)
- [ ] `rag_request_summary` event: `steps_ms`, `model_used`, `degraded`, `prompt/completion tokens`, `estimated vs actual`, `cost_usd`
- [ ] Optional: LangFuse integration

**Acceptance**
- A question with no answer in the corpus: low-score refusal, no `llm` span in the trace
- When a malicious injection doc is retrieved: output contains none of the payloads (phishing link / tampered numbers) and no tag escape happens

---

## M4 · Evaluation + delivery (days 10–12)

- [ ] Grow `evals/golden_set.json` to 30–50 cases: answerable 50% / no_answer 20% / unauthorized + cross_tenant 15% / injection 15%
- [ ] `evals/run_eval.py`: keyword scoring + citation accuracy + retrieval-level leak assertions + (optional) LLM-as-judge faithfulness
- [ ] Compare with `evals/baselines/latest.json`: success rate, refusal accuracy, leak count (must be 0), P50/P95 latency, total cost
- [ ] README: architecture diagram, measured metrics, threshold curve, design trade-offs

**Acceptance**
- `python -m evals.run_eval` prints a comparison report in one command; unauthorized leaks = 0; successful injections = 0

---

## Non-goals

- Real SSO / user management (JWTs are minted by a script, simulating an upstream gateway)
- Sharing breaker state across processes (each worker is independent; trade-off explained in the README)
- Production-grade PDF parsing (mock docs are mainly Markdown)
