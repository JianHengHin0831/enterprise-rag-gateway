"""Query rewrite with the small model (route `rewrite`).

Only multi-turn requests (to resolve "it" / "that" against the history) and very short
queries (to make them self-contained) are rewritten; everything else uses the raw query.
Rewriting is best effort: any LLM error, a timeout, an exhausted budget or an empty /
oversized output falls back to the raw query, and the result says so (`rewritten=False`).
The rewritten text is used only as a search query and question text, never as identity or
instructions.
"""

import asyncio
import logging
from dataclasses import dataclass

from app.cost.budget import BudgetGuard
from app.cost.tokens import estimate_messages_tokens
from app.gateway.base import ChatMessage, LLMError
from app.gateway.router import ModelRouter
from app.observability.logging import log_event
from app.observability.trace import Span
from app.rag.prompts import PromptTemplate
from app.schemas import ChatRequest, Identity

logger = logging.getLogger("rag.rewrite")

MAX_REWRITE_CHARS = 2000


@dataclass(frozen=True)
class RewriteResult:
    query: str
    rewritten: bool  # True only when the model produced the query
    attempted: bool


class QueryRewriter:
    def __init__(
        self,
        *,
        router: ModelRouter,
        prompt: PromptTemplate,
        timeout_s: float,
        min_words: int,
        budget: BudgetGuard | None = None,
    ) -> None:
        self._router = router
        self._prompt = prompt
        self._timeout_s = timeout_s
        self._min_words = min_words
        self._budget = budget

    @property
    def prompt_version(self) -> int:
        return self._prompt.version

    def needs_rewrite(self, request: ChatRequest) -> bool:
        return bool(request.history) or len(request.query.split()) < self._min_words

    def build_messages(self, request: ChatRequest) -> list[ChatMessage]:
        history = "\n".join(f"{turn.role}: {turn.content}" for turn in request.history)
        return [
            ChatMessage(role="system", content=self._prompt.render_system()),
            ChatMessage(
                role="user",
                content=self._prompt.render_user(history=history or "(none)", query=request.query),
            ),
        ]

    async def rewrite(self, request: ChatRequest, identity: Identity) -> RewriteResult:
        raw = RewriteResult(query=request.query, rewritten=False, attempted=False)
        if not self.needs_rewrite(request):
            return raw

        messages = self.build_messages(request)
        reservation = None
        if self._budget is not None:
            estimate = estimate_messages_tokens(messages) + self._router.max_output_tokens(
                "rewrite"
            )
            reservation = await self._budget.reserve(identity.tenant_id, estimate)
            if not reservation.accepted:
                log_event(logger, "rewrite_skipped", reason="budget")
                return RewriteResult(query=request.query, rewritten=False, attempted=True)

        actual = 0
        async with Span("rewrite", turns=len(request.history)) as span:
            try:
                async with asyncio.timeout(self._timeout_s):
                    routed = await self._router.chat("rewrite", messages)
                usage = routed.response.usage
                actual = usage.prompt_tokens + usage.completion_tokens
                span.set("model_used", routed.model_key)
                span.set("degraded", routed.degraded)
                text = _clean(routed.response.text)
            except (LLMError, TimeoutError) as exc:
                span.set("fallback", "raw_query")
                span.set("error_type", type(exc).__name__)
                text = None
            finally:
                if reservation is not None and self._budget is not None:
                    await self._budget.settle(reservation, actual)
            span.set("rewritten", text is not None)

        if text is None:
            return RewriteResult(query=request.query, rewritten=False, attempted=True)
        return RewriteResult(query=text, rewritten=True, attempted=True)


def _clean(text: str) -> str | None:
    """First non-empty line, without wrapping quotes; None if unusable."""
    for line in text.strip().splitlines():
        line = line.strip().strip('"').strip()
        if line:
            return line if len(line) <= MAX_REWRITE_CHARS else None
    return None
