"""Last-resort local provider: no network, fixed safe answer. Always marks the answer as
not cacheable upstream (see app/cache/keys.py::should_cache)."""

from collections.abc import AsyncIterator

from app.gateway.base import BaseLLMProvider, ChatMessage, LLMResponse, StreamChunk, Usage

FALLBACK_TEXT = (
    "The service is busy and cannot generate an answer right now. Please try again later, "
    "or contact the relevant department if it is urgent."
)


class TemplateProvider(BaseLLMProvider):
    name = "local"

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
    ) -> LLMResponse:
        return LLMResponse(
            text=FALLBACK_TEXT, model=model, provider=self.name, usage=Usage(), finish_reason="stop"
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
    ) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(delta=FALLBACK_TEXT)
        yield StreamChunk(usage=Usage(), finish_reason="stop")
