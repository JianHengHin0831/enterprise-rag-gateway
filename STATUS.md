# Status

**Current milestone:** M2 — multi-model gateway + breaker/fallback + budget + cache + SSE
**Last updated:** 2026-11-19

## Done
- M0 skeleton: pure-logic modules (auth / ACL filter / sanitizer / trace / circuit breaker / cache key / citations / injection wrapping / pricing / prompt loader) implemented with unit tests.
- Whole repository is English-only (code, comments, data, prompts, docs).
- Demo console UI at `/ui/` (persona switcher via dev-only `/dev/token`, health/ready panel, chat with citation and status badges).

- M1 items 1–4:
  - Mock corpus: 17 docs (acme_corp 10, beta_inc 7) across all / eng / hr / sales / finance; 2 injection docs (`acme_pantry_guide`, `beta_it_faq`) are flagged `suspicious` at ingest.
  - `app/rag/embeddings.py`: `BgeM3Embedder` (FlagEmbedding, dense 1024-d + sparse lexical weights, optional extra `.[embed]`) and deterministic `FakeEmbedder`; selected by `RAG_EMBEDDER`.
  - `app/rag/store.py`: collection with named vectors `dense` (cosine) + `sparse`, payload indexes on `tenant_id` (`is_tenant`), `allowed_roles`, `doc_id`, `index_version`; refuses a dense-size mismatch unless `--recreate`.
  - `app/rag/ingest.py`: layout guard (dir must match `tenant_id` / `allowed_roles`), duplicate doc_id and unknown tenant checks, deterministic point ids, stale-chunk cleanup on re-ingest, then bump `index_version:{tenant}` in Redis.

- M1 items 5–8 (M1 complete, acceptance green):
  - `HybridRetriever.search`: dense + sparse prefetch, ACL filter on each prefetch and top level, RRF fusion. `tests/test_retriever_acl.py` covers all 12 tenant x role identities.
  - `OpenAICompatProvider.chat` (non-streaming; 429/5xx/timeout -> retryable, 4xx/malformed -> non-retryable, `Retry-After` parsed). Providers registered from `models.yaml` only when their key is set (`app/gateway/factory.py`); `ModelRouter.chat` (M1) calls the first configured model of the chain once.
  - `/v1/chat`: JWT -> trace -> retrieve -> `wrap_documents` -> router -> `answer + citations + trace_id`. LLM errors -> 502, Qdrant errors -> 503, both with `trace_id`. The `retrieve` span logs `retrieved_doc_ids`.
  - `/ready`: Qdrant `/readyz` + collection present + Redis PING.
  - `tests/test_m1_acceptance.py` (HTTP route + real pipeline on in-memory Qdrant) and `scripts/check_acl.py` (live leak check against the running Qdrant).

- M2 items 1–4:
  - `AnthropicProvider.chat` (raw HTTP on the shared client, `x-api-key` + `anthropic-version`): system messages -> top-level `system`, only `text` blocks form the answer, usage `input_tokens` (+ cache fields) / `output_tokens` mapped, `temperature` not sent (current Claude models reject non-default sampling), `stop_reason: refusal` -> `ContentRefusalError`. Registered by `build_providers` when `RAG_ANTHROPIC_API_KEY` is set. Shared HTTP error mapping lives in `app/gateway/providers/_http.py`.
  - Error taxonomy (`app/gateway/base.py`): `ProviderTimeoutError` / `RateLimitedError` / `RetryableLLMError` (retry, fall back, count toward the breaker) vs `NonRetryableLLMError` / `ContentRefusalError` (raised at once, breaker untouched; a half-open probe slot is released via `CircuitBreaker.release`). OpenAI-compatible `finish_reason: content_filter` -> refusal.
  - `ModelRouter.chat`: per-model retries with exponential backoff + full jitter, `Retry-After` honoured (a hint above `max_delay_s` falls back immediately), overall `request_deadline_s` (attempt timeouts capped, no sleep across it, remote models skipped after it; the local template is exempt). All models failing -> `AllModelsFailedError` -> HTTP 503; non-retryable -> 502.
  - Breakers keyed by `provider:model` (model keys pointing at the same upstream model share one). `degraded=true` whenever the answer comes from anything after the first available model; `model_used`, `degraded` and `attempts` are on the `llm` span, `model_used` / `degraded` on the response. Events `llm_attempt_failed`, `llm_model_skipped`, `llm_degraded` log error type + status only.
  - Acceptance `tests/test_router_fallback.py` (5x 429 -> breaker open, 6th call skips primary) unskipped and green; `tests/test_router_resilience.py`, `tests/test_anthropic_provider.py`.

## In progress
- (none)

## Next
- M2 items 5–8: SSE streaming (provider `stream` for both adapters + router pre-first-token fallback), budget, L1/L2 cache, query rewrite.

## Known gaps
- Claude Sonnet / Haiku 5.5 think adaptively by default and thinking tokens count toward `max_tokens`; with `max_output_tokens: 1024` / `256` a long thought could truncate the answer. The adapter sends no `thinking` / effort setting yet; revisit if answers come back cut off (`finish_reason == "max_tokens"`).
- Content refusals are raised (502), not routed to another model.
- No reranker / low-score gate yet (M3): the first 5 fused hits go to the LLM; only an empty retrieval is refused without an LLM call.
- `history` is ignored until query rewrite exists (M2).
- The Docker image installs without the `[embed]` extra, so `RAG_EMBEDDER=bge-m3` only works for a locally run API; the full-stack image needs a CPU-torch install step.
- Reranker, budget and L1/L2 cache are `TODO(Mx)` stubs.
- Prices in `configs/models.yaml` are `null`; fill them in from each provider's current price page.
