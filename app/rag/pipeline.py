"""End-to-end RAG orchestration for one request.

Order (each step inside a `Span`):
  1. budget.reserve(estimate)                     -> 429 / downgrade        (M2)
  2. screen_text(query)                           -> risk flag in trace     (M1, in the API)
  3. rewrite (only multi-turn / short query; on failure use raw query)    (M2)
  4. cache lookup with build_cache_key(rewritten) -> hit: return           (M2)
  5. retriever.search(query, identity)            -> ACL at DB level       (M1)
  6. reranker.rerank + passes_gate                -> refuse w/o LLM        (M3)
  7. wrap_documents + render prompt -> router.chat/stream("answer")       (M1/M2)
  8. check_citations -> disclaimer on mismatch                            (M3)
  9. should_cache -> cache.set; budget.settle; audit; rag_request_summary  (M2/M3)

M1 implements 5 and 7 (non-streaming) and maps `[n]` markers to citations. Until the
reranker exists (M3), the first `context_top_n` fused hits go to the LLM, and an empty
retrieval result is refused without an LLM call.
"""

import logging
from typing import Protocol

from app.gateway.base import ChatMessage
from app.gateway.router import ModelRouter
from app.observability.logging import log_event
from app.observability.trace import Span, current_trace_id, steps_ms
from app.rag.citations import check_citations
from app.rag.prompts import PromptTemplate
from app.schemas import ChatRequest, ChatResponse, Citation, Identity, RetrievedChunk
from app.security.injection import wrap_documents

logger = logging.getLogger("rag.pipeline")


class Retriever(Protocol):
    async def search(self, query: str, identity: Identity, top_k: int) -> list[RetrievedChunk]: ...


class RagPipeline:
    def __init__(
        self,
        *,
        retriever: Retriever,
        router: ModelRouter,
        prompt: PromptTemplate,
        retrieve_top_k: int,
        context_top_n: int,
    ) -> None:
        self._retriever = retriever
        self._router = router
        self._prompt = prompt
        self._retrieve_top_k = retrieve_top_k
        self._context_top_n = context_top_n

    @property
    def refusal_text(self) -> str:
        return self._prompt.refusal_text or ""

    async def answer(self, request: ChatRequest, identity: Identity) -> ChatResponse:
        # TODO(M2): multi-turn history goes through query rewrite; M1 answers `query` alone.
        query = request.query

        async with Span("retrieve", top_k=self._retrieve_top_k) as span:
            hits = await self._retriever.search(query, identity, self._retrieve_top_k)
            span.set("hits", len(hits))
            span.set("retrieved_doc_ids", sorted({h.doc_id for h in hits}))
            span.set("suspicious_hits", sum(h.suspicious for h in hits))

        contexts = hits[: self._context_top_n]
        if not contexts:
            return self._finish(
                identity, ChatResponse(trace_id=_tid(), answer=self.refusal_text, refused=True)
            )

        wrapped = wrap_documents(contexts)
        messages = [
            ChatMessage(role="system", content=self._prompt.render_system(doc_tag=wrapped.tag)),
            ChatMessage(
                role="user",
                content=self._prompt.render_user(documents=wrapped.text, query=query),
            ),
        ]

        async with Span("llm", task="answer", contexts=len(contexts)) as span:
            routed = await self._router.chat("answer", messages)
            span.set("model_used", routed.model_key)
            span.set("degraded", routed.degraded)
            span.set("attempts", routed.attempts)
            span.set("prompt_tokens", routed.response.usage.prompt_tokens)
            span.set("completion_tokens", routed.response.usage.completion_tokens)

        text = routed.response.text.strip()
        check = check_citations(text, len(contexts))
        citations = [
            Citation(
                index=n,
                doc_id=contexts[n - 1].doc_id,
                chunk_index=contexts[n - 1].chunk_index,
                title=contexts[n - 1].title,
            )
            for n in check.cited
            if n not in check.invalid
        ]
        refused = bool(self.refusal_text) and self.refusal_text in text
        # TODO(M3): append DISCLAIMER and record the metric on citation_mismatch.
        response = ChatResponse(
            trace_id=_tid(),
            answer=text,
            citations=[] if refused else citations,
            refused=refused,
            model_used=routed.model_key,
            degraded=routed.degraded,
            citation_mismatch=check.mismatch,
        )
        return self._finish(identity, response)

    def _finish(self, identity: Identity, response: ChatResponse) -> ChatResponse:
        log_event(
            logger,
            "chat_response",
            tenant_id=identity.tenant_id,
            role=identity.role.value,
            user_id=identity.user_id,
            refused=response.refused,
            model_used=response.model_used,
            degraded=response.degraded,
            citations=[c.doc_id for c in response.citations],
            citation_mismatch=response.citation_mismatch,
            steps_ms=steps_ms(),
        )
        return response


def _tid() -> str:
    return current_trace_id() or ""
