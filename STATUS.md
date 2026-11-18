# Status

**Current milestone:** M1 — core path + access control + logging
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

## In progress
- (none)

## Next
- M1 item 5: `HybridRetriever.search` (prefetch dense + sparse with the ACL filter on each, RRF fusion) and `tests/test_retriever_acl.py`.

## Known gaps
- `/v1/chat` currently returns 501 (pipeline not wired up).
- Provider adapters, retriever, reranker, budget and L1/L2 cache are `TODO(Mx)` stubs.
- Prices in `configs/models.yaml` are `null`; fill them in from each provider's current price page.
