"""Provider-agnostic LLM interface and error taxonomy.

Error taxonomy (decides what the router does):

| Error                    | Cause                                   | Retry / fallback | Breaker  |
|--------------------------|-----------------------------------------|------------------|----------|
| ProviderTimeoutError     | timeout, request deadline               | yes              | failure  |
| RateLimitedError         | HTTP 429 (honours Retry-After)          | yes              | failure  |
| RetryableLLMError        | 408/409/5xx/529, connection errors      | yes              | failure  |
| NonRetryableLLMError     | 400/401/403/404/422, malformed response | no, raise        | untouched|
| ContentRefusalError      | model or provider content refusal       | no, raise        | untouched|
| AllModelsFailedError     | router: every model of the chain failed | (final error)    | n/a      |

A non-retryable error is our bug or a policy decision, not a provider outage, so it must not
open the breaker or burn the fallback chain.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Literal

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMResponse(BaseModel):
    text: str
    model: str
    provider: str
    usage: Usage
    finish_reason: str | None = None


class StreamChunk(BaseModel):
    delta: str = ""
    usage: Usage | None = None  # set on the final chunk when the provider reports it
    finish_reason: str | None = None


class LLMError(Exception):
    """Base error. `retryable` decides retry/fallback; only retryable errors count toward
    the circuit breaker (a 400 is our bug, not the provider's outage)."""

    retryable: bool = False

    def __init__(
        self, message: str, *, status_code: int | None = None, retry_after_s: float | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_s = retry_after_s


class RetryableLLMError(LLMError):
    retryable = True


class RateLimitedError(RetryableLLMError):
    pass


class ProviderTimeoutError(RetryableLLMError):
    pass


class NonRetryableLLMError(LLMError):
    retryable = False


class ContentRefusalError(NonRetryableLLMError):
    """The model or the provider's content filter declined to answer."""


class AllModelsFailedError(RetryableLLMError):
    """Raised by the router when every model of the chain failed with a retryable error, had
    its breaker open, or ran out of the request deadline. The caller may retry later."""

    def __init__(self, message: str, *, last_error: LLMError | None = None) -> None:
        super().__init__(
            message,
            status_code=last_error.status_code if last_error else None,
            retry_after_s=last_error.retry_after_s if last_error else None,
        )
        self.last_error = last_error


_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


def error_from_status(
    status_code: int, message: str, retry_after_s: float | None = None
) -> LLMError:
    if status_code == 429:
        return RateLimitedError(message, status_code=429, retry_after_s=retry_after_s)
    if status_code in _RETRYABLE_STATUS:
        return RetryableLLMError(message, status_code=status_code, retry_after_s=retry_after_s)
    return NonRetryableLLMError(message, status_code=status_code)


class BaseLLMProvider(ABC):
    """One adapter per API dialect (OpenAI-compatible, Anthropic, local template...)."""

    name: str

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
    ) -> LLMResponse: ...

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        max_tokens: int,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
    ) -> AsyncIterator[StreamChunk]: ...
