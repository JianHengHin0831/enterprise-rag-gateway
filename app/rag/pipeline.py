"""End-to-end RAG orchestration for one request.

Order (each step inside a `Span`):
  1. rewrite (only multi-turn / short query; on failure use raw query)     (M2)
  2. cache lookup with build_cache_key(rewritten) -> hit: return           (M2)
  3. retriever.search(query, identity)            -> ACL at DB level       (M1)
  4. reranker.rerank + passes_gate                -> refuse w/o LLM        (M3)
  5. wrap_documents + render prompt
  6. budget.reserve(estimate)                     -> 429 / downgrade        (M2)
  7. router.chat / router.open_stream ("answer")                            (M1/M2)
  8. check_citations on the full text -> disclaimer + metric on mismatch   (M3)
  9. budget.settle; should_cache -> cache.set; audit; rag_request_summary  (M2/M3)

(`screen_text(query)` runs in the API layer before the pipeline.)

Budget: the reservation covers the prompt estimate plus the largest `max_output_tokens` of
the route's chain. When it is refused, `on_exceeded: downgrade` retries the reservation for
the small-model route (`rewrite` chain, smaller outputs) and marks the answer degraded;
otherwise, or when even that does not fit, BudgetExceededError (HTTP 429).

Caching: only single-turn requests, or multi-turn ones whose rewrite succeeded, are
cacheable (a raw follow-up like "and for interns?" means nothing on its own). Cache hits are
returned with `cache_hit=True`, `cost_usd=0`.

Rerank + gate: the `retrieve_top_k` fused hits are reranked and the best `context_top_n` kept.
If the top rerank score is below `rerank_threshold` (or nothing was retrieved) the fixed
refusal text is returned without an LLM call. Low-score refusals are not cached, so a
recalibrated threshold takes effect at once. Without a reranker the first `context_top_n`
fused hits go to the LLM (no gate).

Citations: the check runs on the complete answer (for streams: after the last delta). On a
mismatch the DISCLAIMER is appended (streams get it as one extra `delta`), the metric
`rag_citation_mismatch_total` is incremented and the answer is never cached.
"""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from app.cache.keys import build_cache_key, should_cache
from app.cache.layered import LayeredCache
from app.cost.budget import BudgetExceededError, BudgetGuard, Reservation
from app.cost.pricing import compute_cost_usd
from app.cost.tokens import estimate_messages_tokens
from app.gateway.base import ChatMessage, Usage
from app.gateway.router import ModelRouter, Task
from app.observability import metrics
from app.observability.logging import log_event
from app.observability.trace import Span, current_trace_id, steps_ms
from app.rag.citations import DISCLAIMER, check_citations
from app.rag.prompts import PromptTemplate
from app.rag.reranker import Reranker, passes_gate
from app.rag.rewrite import QueryRewriter, RewriteResult
from app.schemas import ChatRequest, ChatResponse, Citation, Identity, RetrievedChunk
from app.security.injection import wrap_documents

logger = logging.getLogger("rag.pipeline")


class Retriever(Protocol):
    async def search(self, query: str, identity: Identity, top_k: int) -> list[RetrievedChunk]: ...


@dataclass(frozen=True)
class StreamEvent:
    """One server-sent event: `meta` (model chosen, first token in hand), `delta` (answer
    text), `done` (the final ChatResponse, citations included)."""

    kind: Literal["meta", "delta", "done"]
    data: dict[str, Any]


@dataclass
class _Plan:
    rewrite: RewriteResult
    cache_key: str | None = None
    response: ChatResponse | None = None  # set when no LLM call is needed (hit / refusal)
    contexts: list[RetrievedChunk] = field(default_factory=list)
    messages: list[ChatMessage] = field(default_factory=list)
    task: Task = "answer"
    downgraded: bool = False
    reservation: Reservation | None = None
    refusal_reason: Literal["no_results", "low_score"] | None = None


class RagPipeline:
    def __init__(
        self,
        *,
        retriever: Retriever,
        router: ModelRouter,
        prompt: PromptTemplate,
        retrieve_top_k: int,
        context_top_n: int,
        budget: BudgetGuard | None = None,
        cache: LayeredCache | None = None,
        rewriter: QueryRewriter | None = None,
        reranker: Reranker | None = None,
        rerank_threshold: float = 0.0,
    ) -> None:
        self._retriever = retriever
        self._router = router
        self._prompt = prompt
        self._retrieve_top_k = retrieve_top_k
        self._context_top_n = context_top_n
        self._budget = budget
        self._cache = cache
        self._rewriter = rewriter
        self._reranker = reranker
        self._rerank_threshold = rerank_threshold

    @property
    def refusal_text(self) -> str:
        return self._prompt.refusal_text or ""

    # ---------- public entry points ----------

    async def answer(self, request: ChatRequest, identity: Identity) -> ChatResponse:
        plan = await self._prepare(request, identity)
        if plan.response is not None:
            return self._finish(identity, plan, plan.response)

        actual = 0
        try:
            async with Span("llm", task=plan.task, contexts=len(plan.contexts)) as span:
                routed = await self._router.chat(plan.task, plan.messages)
                usage = routed.response.usage
                actual = usage.prompt_tokens + usage.completion_tokens
                _llm_span_attrs(span, routed.model_key, routed.degraded, routed.attempts, usage)
        finally:
            await self._settle(plan, actual)

        response = self._build(
            plan, routed.response.text, routed.model_key, routed.degraded, routed.response.usage
        )
        await self._store(plan, response)
        return self._finish(identity, plan, response)

    async def stream(self, request: ChatRequest, identity: Identity) -> AsyncIterator[StreamEvent]:
        """Async generator of StreamEvents. Everything up to and including the first token
        happens before the first (`meta`) event, so the API layer can still answer with a
        plain HTTP error when it fails; later failures propagate after `meta` was sent."""
        plan = await self._prepare(request, identity)
        if plan.response is not None:
            response = self._finish(identity, plan, plan.response)
            yield _meta(response.model_used, response.degraded, response.cache_hit)
            if response.answer:
                yield StreamEvent("delta", {"text": response.answer})
            yield StreamEvent("done", response.model_dump())
            return

        started = False
        settled = False
        try:
            async with Span(
                "llm", task=plan.task, contexts=len(plan.contexts), stream=True
            ) as span:
                routed = await self._router.open_stream(plan.task, plan.messages)
                started = True
                degraded = routed.degraded or plan.downgraded
                yield _meta(routed.model_key, degraded, False)

                parts: list[str] = []
                usage = Usage()
                async for chunk in routed.chunks:
                    if chunk.delta:
                        parts.append(chunk.delta)
                        yield StreamEvent("delta", {"text": chunk.delta})
                    if chunk.usage is not None:
                        usage = chunk.usage
                _llm_span_attrs(span, routed.model_key, routed.degraded, routed.attempts, usage)

            await self._settle(plan, usage.prompt_tokens + usage.completion_tokens)
            settled = True
            response = self._build(plan, "".join(parts), routed.model_key, routed.degraded, usage)
            if response.citation_mismatch:
                # The deltas are already out; send the disclaimer so the client's text
                # ends the same way as the `done` answer.
                yield StreamEvent("delta", {"text": DISCLAIMER})
            await self._store(plan, response)
            yield StreamEvent("done", self._finish(identity, plan, response).model_dump())
        finally:
            if not settled:
                # Before the first token nothing was generated: refund. After it the real
                # usage is unknown, so the estimate stays booked (conservative).
                reservation = plan.reservation
                await self._settle(plan, reservation.tokens if started and reservation else 0)

    # ---------- stages ----------

    async def _prepare(self, request: ChatRequest, identity: Identity) -> _Plan:
        if self._rewriter is not None:
            rewrite = await self._rewriter.rewrite(request, identity)
        else:
            rewrite = RewriteResult(query=request.query, rewritten=False, attempted=False)
        plan = _Plan(rewrite=rewrite)
        query = rewrite.query

        cacheable = not request.history or rewrite.rewritten
        if self._cache is not None and cacheable:
            async with Span("cache_lookup") as span:
                version = await self._cache.index_version(identity.tenant_id)
                if version is not None:
                    plan.cache_key = build_cache_key(
                        tenant_id=identity.tenant_id,
                        role=identity.role.value,
                        prompt_version=self._prompt.version,
                        rewrite_prompt_version=(
                            self._rewriter.prompt_version if self._rewriter else 0
                        ),
                        index_version=version,
                        query=query,
                    )
                    plan.response = await self._cache.get(plan.cache_key)
                span.set("hit", plan.response is not None)
            if plan.response is not None:
                return plan

        async with Span("retrieve", top_k=self._retrieve_top_k) as span:
            hits = await self._retriever.search(query, identity, self._retrieve_top_k)
            span.set("hits", len(hits))
            span.set("retrieved_doc_ids", sorted({h.doc_id for h in hits}))
            span.set("suspicious_hits", sum(h.suspicious for h in hits))

        if not hits:
            await self._refuse(plan, "no_results")
            return plan

        if self._reranker is None:
            plan.contexts = hits[: self._context_top_n]
        else:
            async with Span("rerank", reranker=self._reranker.name, candidates=len(hits)) as span:
                ranked = await self._reranker.rerank(query, hits, self._context_top_n)
                passed = passes_gate(ranked, self._rerank_threshold)
                top = ranked[0].rerank_score if ranked else None
                span.set("top_score", None if top is None else round(top, 4))
                span.set("threshold", self._rerank_threshold)
                span.set("passed", passed)
                span.set("context_refs", [c.ref for c in ranked])
            metrics.inc("rag_rerank_gate_total", decision="passed" if passed else "refused")
            if not passed:
                await self._refuse(plan, "low_score")
                return plan
            plan.contexts = ranked

        wrapped = wrap_documents(plan.contexts)
        plan.messages = [
            ChatMessage(role="system", content=self._prompt.render_system(doc_tag=wrapped.tag)),
            ChatMessage(
                role="user",
                content=self._prompt.render_user(documents=wrapped.text, query=query),
            ),
        ]
        await self._reserve(plan, identity)
        return plan

    async def _refuse(self, plan: _Plan, reason: Literal["no_results", "low_score"]) -> None:
        """Fixed refusal without an LLM call."""
        plan.refusal_reason = reason
        plan.response = ChatResponse(
            trace_id=_tid(), answer=self.refusal_text, refused=True, cost_usd=0.0
        )
        metrics.inc("rag_refusals_total", reason=reason)
        if reason == "no_results":  # deterministic per index_version; low_score depends on
            await self._store(plan, plan.response)  # the threshold, which is not in the key

    async def _reserve(self, plan: _Plan, identity: Identity) -> None:
        if self._budget is None:
            return
        prompt_tokens = estimate_messages_tokens(plan.messages)
        async with Span("budget") as span:
            reservation = await self._budget.reserve(
                identity.tenant_id, prompt_tokens + self._router.max_output_tokens("answer")
            )
            if (
                not reservation.accepted
                and self._budget.on_exceeded == "downgrade"
                and self._router.available_chain("rewrite")
            ):
                reservation = await self._budget.reserve(
                    identity.tenant_id, prompt_tokens + self._router.max_output_tokens("rewrite")
                )
                if reservation.accepted:
                    plan.task, plan.downgraded = "rewrite", True
            span.set("estimated_tokens", reservation.tokens)
            span.set("accepted", reservation.accepted)
            span.set("downgraded", plan.downgraded)
        if not reservation.accepted:
            raise BudgetExceededError(identity.tenant_id)
        plan.reservation = reservation

    async def _settle(self, plan: _Plan, actual_tokens: int) -> None:
        if self._budget is not None and plan.reservation is not None:
            await self._budget.settle(plan.reservation, actual_tokens)
            plan.reservation = None  # settle exactly once

    async def _store(self, plan: _Plan, response: ChatResponse) -> None:
        if self._cache is None or plan.cache_key is None:
            return
        if should_cache(response, fallback_model_keys=self._router.fallback_model_keys()):
            await self._cache.set(plan.cache_key, response)

    def _build(
        self, plan: _Plan, text: str, model_key: str, degraded: bool, usage: Usage
    ) -> ChatResponse:
        text = text.strip()
        contexts = plan.contexts
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
        mismatch = check.mismatch and not refused
        if refused:
            metrics.inc("rag_refusals_total", reason="model")
        if mismatch:
            text += DISCLAIMER
            metrics.inc("rag_citation_mismatch_total")
            log_event(
                logger,
                "citation_mismatch",
                level=logging.WARNING,
                invalid=list(check.invalid),
                num_contexts=len(contexts),
                model_used=model_key,
            )
        return ChatResponse(
            trace_id=_tid(),
            answer=text,
            citations=[] if refused else citations,
            refused=refused,
            model_used=model_key,
            degraded=degraded or plan.downgraded,
            citation_mismatch=mismatch,
            cost_usd=compute_cost_usd(usage, self._router.spec(model_key)),
        )

    def _finish(self, identity: Identity, plan: _Plan, response: ChatResponse) -> ChatResponse:
        log_event(
            logger,
            "chat_response",
            tenant_id=identity.tenant_id,
            role=identity.role.value,
            user_id=identity.user_id,
            refused=response.refused,
            refusal_reason=plan.refusal_reason,
            cache_hit=response.cache_hit,
            rewritten=plan.rewrite.rewritten,
            model_used=response.model_used,
            degraded=response.degraded,
            downgraded=plan.downgraded,
            cost_usd=response.cost_usd,
            citations=[c.doc_id for c in response.citations],
            citation_mismatch=response.citation_mismatch,
            steps_ms=steps_ms(),
        )
        return response


def _meta(model_used: str | None, degraded: bool, cache_hit: bool) -> StreamEvent:
    return StreamEvent(
        "meta",
        {
            "trace_id": _tid(),
            "model_used": model_used,
            "degraded": degraded,
            "cache_hit": cache_hit,
        },
    )


def _llm_span_attrs(
    span: Span, model_key: str, degraded: bool, attempts: int, usage: Usage
) -> None:
    span.set("model_used", model_key)
    span.set("degraded", degraded)
    span.set("attempts", attempts)
    span.set("prompt_tokens", usage.prompt_tokens)
    span.set("completion_tokens", usage.completion_tokens)


def _tid() -> str:
    return current_trace_id() or ""
