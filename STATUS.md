# Status

**Current milestone:** M1 — core path + access control + logging
**Last updated:** 2026-11-18

## Done
- M0 skeleton: pure-logic modules (auth / ACL filter / sanitizer / trace / circuit breaker / cache key / citations / injection wrapping / pricing / prompt loader) implemented with unit tests.
- Whole repository is English-only (code, comments, data, prompts, docs).
- Demo console UI at `/ui/` (persona switcher via dev-only `/dev/token`, health/ready panel, chat with citation and status badges).

## In progress
- (none)

## Next
- First M1 item: grow the mock docs, integrate bge-m3, implement the ingest upsert.

## Known gaps
- `/v1/chat` currently returns 501 (pipeline not wired up).
- Provider adapters, retriever, reranker, budget and L1/L2 cache are `TODO(Mx)` stubs.
- Prices in `configs/models.yaml` are `null`; fill them in from each provider's current price page.
