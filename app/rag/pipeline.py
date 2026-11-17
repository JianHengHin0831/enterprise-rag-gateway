"""End-to-end RAG orchestration for one request.

Order (each step inside a `Span`):
  1. budget.reserve(estimate)                     -> 429 / downgrade        (M2)
  2. screen_text(query)                           -> risk flag in trace     (M1)
  3. rewrite (only multi-turn / short query; on failure use raw query)    (M2)
  4. cache lookup with build_cache_key(rewritten) -> hit: return           (M2)
  5. retriever.search(query, identity)            -> ACL at DB level       (M1)
  6. reranker.rerank + passes_gate                -> refuse w/o LLM        (M3)
  7. wrap_documents + render prompt -> router.chat/stream("answer")       (M1/M2)
  8. check_citations -> disclaimer on mismatch                            (M3)
  9. should_cache -> cache.set; budget.settle; audit; rag_request_summary  (M2/M3)
"""

from app.schemas import ChatRequest, ChatResponse, Identity


class RagPipeline:
    async def answer(self, request: ChatRequest, identity: Identity) -> ChatResponse:
        raise NotImplementedError("RagPipeline.answer — milestone M1")
