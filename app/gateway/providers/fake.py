"""Scriptable fake provider for tests. Never used outside tests/evals dry runs."""

from collections import deque
from collections.abc import AsyncIterator, Iterable

from app.gateway.base import (
    BaseLLMProvider,
    ChatMessage,
    LLMError,
    LLMResponse,
    StreamChunk,
    Usage,
)

Outcome = str | LLMError


class FakeLLM(BaseLLMProvider):
    """Each call consumes the next scripted outcome: a str is returned as the answer, an
    LLMError instance is raised. When the script is exhausted, `default` is returned."""

    def __init__(
        self, name: str = "fake", script: Iterable[Outcome] = (), default: str = "ok [1]"
    ) -> None:
        self.name = name
        self._script: deque[Outcome] = deque(script)
        self._default = default
        self.calls: list[str] = []  # model names, in call order

    def _next(self, model: str) -> str:
        self.calls.append(model)
        outcome = self._script.popleft() if self._script else self._default
        if isinstance(outcome, LLMError):
            raise outcome
        return outcome

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
    ) -> LLMResponse:
        text = self._next(model)
        return LLMResponse(
            text=text,
            model=model,
            provider=self.name,
            usage=Usage(
                prompt_tokens=sum(len(m.content) for m in messages) // 2,
                completion_tokens=len(text) // 2,
            ),
            finish_reason="stop",
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
        text = self._next(model)
        for i in range(0, len(text), 4):
            yield StreamChunk(delta=text[i : i + 4])
        yield StreamChunk(usage=Usage(completion_tokens=len(text) // 2), finish_reason="stop")
