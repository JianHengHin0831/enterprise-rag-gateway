# Status

**Current milestone:** M2 — multi-model gateway + breaker/fallback + budget + cache + SSE
**Last updated:** 2026-11-18

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

## In progress
- (none)

## Next
- M2: router retry/backoff/breaker/fallback, Anthropic adapter, SSE streaming, budget, L1/L2 cache, query rewrite.

## Known gaps
- No retry/fallback yet: a failing primary model returns 502 (M2). Anthropic adapter is M2, so the answer route uses the first configured OpenAI-compatible model, or the template fallback when no key is set.
- No reranker / low-score gate yet (M3): the first 5 fused hits go to the LLM; only an empty retrieval is refused without an LLM call.
- `history` is ignored until query rewrite exists (M2).
- The Docker image installs without the `[embed]` extra, so `RAG_EMBEDDER=bge-m3` only works for a locally run API; the full-stack image needs a CPU-torch install step.
- Reranker, budget and L1/L2 cache are `TODO(Mx)` stubs.
- Prices in `configs/models.yaml` are `null`; fill them in from each provider's current price page.
