"""ModelRouter: the single entry point for every LLM call in the app.

`chat(task, messages)`:
- Resolve the fallback chain for a task (`rewrite` / `answer`) from configs/models.yaml,
  minus models whose provider is not configured (`available_chain`).
- Per model: consult its CircuitBreaker (one per `provider:model`, shared by every model key
  that points at the same upstream model), call the provider, and retry RETRYABLE errors up
  to `retry.max_attempts` times with exponential backoff + full jitter:
      delay = uniform(0, min(max_delay_s, base_delay_s * 2**attempt))
  A `Retry-After` hint raises the delay to at least that value; a hint longer than
  `max_delay_s` means "do not wait here", so the router moves to the next model at once.
- Everything is bounded by `retry.request_deadline_s`: each attempt's timeout is capped at
  the time left, no backoff sleep may cross the deadline, and once it has passed remote
  models are skipped. The local template provider is exempt so a last-resort answer is
  always possible.
- NON-retryable errors (400/401/403, content refusal, malformed response) are raised at
  once: no retry, no fallback, and the breaker is not touched.
- Serving from any model after the first available one -> `degraded=True` (the static
  template is always degraded). Skipping an unconfigured model is a deployment choice, not a
  degradation.
- If no model produced an answer -> AllModelsFailedError (carries the last error).

Streaming (TODO(M2), SSE item): retry/fallback is allowed only until the first chunk has
been yielded. After that, errors propagate and the API layer emits `event: error`.
"""

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from app.gateway.base import (
    AllModelsFailedError,
    BaseLLMProvider,
    ChatMessage,
    LLMError,
    LLMResponse,
    NonRetryableLLMError,
    ProviderTimeoutError,
)
from app.gateway.circuit_breaker import CircuitBreaker
from app.gateway.providers.template import TemplateProvider
from app.observability.logging import log_event

logger = logging.getLogger("rag.gateway")

Task = Literal["rewrite", "answer"]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    provider: str
    model: str
    input_per_mtok: float | None
    output_per_mtok: float | None
    timeout_s: float
    max_output_tokens: int

    @property
    def breaker_name(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 4.0
    request_deadline_s: float = 15.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RetryPolicy":
        r = config.get("retry", {})
        jitter = r.get("jitter", "full")
        if jitter != "full":
            raise ValueError(f"unsupported retry.jitter '{jitter}' (only 'full')")
        return cls(
            max_attempts=max(1, int(r.get("max_attempts", cls.max_attempts))),
            base_delay_s=float(r.get("base_delay_s", cls.base_delay_s)),
            max_delay_s=float(r.get("max_delay_s", cls.max_delay_s)),
            request_deadline_s=float(r.get("request_deadline_s", cls.request_deadline_s)),
        )


@dataclass(frozen=True)
class RoutedResponse:
    response: LLMResponse
    model_key: str
    degraded: bool
    attempts: int  # provider calls made for this request, across retries and fallbacks


def load_models_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_model_specs(config: dict[str, Any]) -> dict[str, ModelSpec]:
    return {
        key: ModelSpec(
            key=key,
            provider=m["provider"],
            model=m["model"],
            input_per_mtok=m.get("input_per_mtok"),
            output_per_mtok=m.get("output_per_mtok"),
            timeout_s=float(m["timeout_s"]),
            max_output_tokens=int(m["max_output_tokens"]),
        )
        for key, m in config["models"].items()
    }


class ModelRouter:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        providers: dict[str, BaseLLMProvider],
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._config = config
        self._providers = providers
        self._specs = parse_model_specs(config)
        self._retry = RetryPolicy.from_config(config)
        self._clock = clock
        self._sleep = sleep
        self._rng = rng or random.Random()  # noqa: S311 - jitter, not crypto
        cb = config.get("circuit_breaker", {})
        self._breakers: dict[str, CircuitBreaker] = {}
        for spec in self._specs.values():
            if spec.breaker_name not in self._breakers:
                self._breakers[spec.breaker_name] = CircuitBreaker(
                    spec.breaker_name,
                    failure_threshold=cb.get("failure_threshold", 5),
                    recovery_timeout_s=cb.get("recovery_timeout_s", 30),
                    half_open_max_calls=cb.get("half_open_max_calls", 2),
                    clock=clock,
                )

    @property
    def retry_policy(self) -> RetryPolicy:
        return self._retry

    def chain(self, task: Task) -> list[ModelSpec]:
        return [self._specs[k] for k in self._config["routes"][task]]

    def breaker(self, model_key: str) -> CircuitBreaker:
        return self._breakers[self._specs[model_key].breaker_name]

    def available_chain(self, task: Task) -> list[ModelSpec]:
        """The task's chain minus models whose provider is not configured (no API key or
        adapter not implemented yet)."""
        return [spec for spec in self.chain(task) if spec.provider in self._providers]

    def backoff_s(self, attempt: int, retry_after_s: float | None) -> float | None:
        """Delay before retry number `attempt + 1` (attempt is 0-based), or None when the
        provider asked us to wait longer than `max_delay_s` (then fall back instead)."""
        cap = min(self._retry.max_delay_s, self._retry.base_delay_s * (2**attempt))
        delay = self._rng.uniform(0.0, cap) if cap > 0 else 0.0
        if retry_after_s is not None:
            if retry_after_s > self._retry.max_delay_s:
                return None
            delay = max(delay, retry_after_s)
        return delay

    async def chat(self, task: Task, messages: list[ChatMessage]) -> RoutedResponse:
        chain = self.available_chain(task)
        if not chain:
            raise NonRetryableLLMError(f"no configured model for task '{task}'")

        deadline = self._clock() + self._retry.request_deadline_s
        attempts = 0
        last_error: LLMError | None = None

        for position, spec in enumerate(chain):
            provider = self._providers[spec.provider]
            is_local = isinstance(provider, TemplateProvider)
            breaker = self.breaker(spec.key)

            for attempt in range(self._retry.max_attempts):
                remaining = deadline - self._clock()
                if remaining <= 0 and not is_local:
                    self._log_skip(task, spec, "deadline_exceeded")
                    break
                if not breaker.allow_request():
                    self._log_skip(task, spec, f"breaker_{breaker.state.value}")
                    break

                attempts += 1
                timeout_s = spec.timeout_s if is_local else min(spec.timeout_s, remaining)
                try:
                    response = await self._call(provider, spec, messages, timeout_s)
                except LLMError as exc:
                    if not exc.retryable:
                        breaker.release()
                        self._log_failure(task, spec, attempt, exc, retry_in_s=None)
                        raise
                    breaker.record_failure()
                    last_error = exc
                    delay = None
                    if attempt + 1 < self._retry.max_attempts:
                        delay = self.backoff_s(attempt, exc.retry_after_s)
                        if delay is not None and self._clock() + delay >= deadline:
                            delay = None
                    self._log_failure(task, spec, attempt, exc, retry_in_s=delay)
                    if delay is None:
                        break
                    await self._sleep(delay)
                    continue
                except BaseException:
                    # Cancellation or a bug: no verdict on the provider's health.
                    breaker.release()
                    raise

                breaker.record_success()
                degraded = position > 0 or is_local
                if degraded:
                    log_event(
                        logger,
                        "llm_degraded",
                        level=logging.WARNING,
                        task=task,
                        model_key=spec.key,
                        attempts=attempts,
                    )
                return RoutedResponse(
                    response=response, model_key=spec.key, degraded=degraded, attempts=attempts
                )

        raise AllModelsFailedError(
            f"all models failed for task '{task}' after {attempts} attempt(s)",
            last_error=last_error,
        )

    async def _call(
        self,
        provider: BaseLLMProvider,
        spec: ModelSpec,
        messages: list[ChatMessage],
        timeout_s: float,
    ) -> LLMResponse:
        # The provider's own (httpx) timeout is per network operation; this one is a hard
        # wall-clock bound so the request deadline holds whatever the provider does.
        try:
            async with asyncio.timeout(timeout_s):
                return await provider.chat(
                    messages,
                    model=spec.model,
                    max_tokens=spec.max_output_tokens,
                    timeout_s=timeout_s,
                )
        except TimeoutError as exc:
            raise ProviderTimeoutError(
                f"{spec.breaker_name}: no answer within {timeout_s:.2f}s"
            ) from exc

    @staticmethod
    def _log_failure(
        task: str, spec: ModelSpec, attempt: int, exc: LLMError, *, retry_in_s: float | None
    ) -> None:
        # Error type and status only: provider error bodies may echo the prompt.
        log_event(
            logger,
            "llm_attempt_failed",
            level=logging.WARNING,
            task=task,
            model_key=spec.key,
            breaker=spec.breaker_name,
            attempt=attempt + 1,
            error=type(exc).__name__,
            status_code=exc.status_code,
            retryable=exc.retryable,
            retry_in_s=None if retry_in_s is None else round(retry_in_s, 3),
        )

    @staticmethod
    def _log_skip(task: str, spec: ModelSpec, reason: str) -> None:
        log_event(
            logger,
            "llm_model_skipped",
            level=logging.WARNING,
            task=task,
            model_key=spec.key,
            breaker=spec.breaker_name,
            reason=reason,
        )
