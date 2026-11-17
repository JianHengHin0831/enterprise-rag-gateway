"""ModelRouter: the single entry point for every LLM call in the app.

Responsibilities (see ROADMAP M2):
- Resolve the fallback chain for a task (`rewrite` / `answer`) from configs/models.yaml.
- Per model: consult its CircuitBreaker, call the provider, retry RETRYABLE errors with
  exponential backoff + full jitter (honour `retry_after_s`), bounded by a request deadline.
- NON-retryable errors fail immediately and do not touch the breaker.
- On exhaustion / open breaker, move to the next model -> `degraded=True`.
- Streaming: retry/fallback is allowed only until the first chunk has been yielded. After
  that, errors propagate and the API layer emits `event: error`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from app.gateway.base import BaseLLMProvider, ChatMessage, LLMResponse
from app.gateway.circuit_breaker import CircuitBreaker

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


@dataclass(frozen=True)
class RoutedResponse:
    response: LLMResponse
    model_key: str
    degraded: bool
    attempts: int


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
    ) -> None:
        self._config = config
        self._providers = providers
        self._specs = parse_model_specs(config)
        cb = config.get("circuit_breaker", {})
        self._breakers = {
            key: CircuitBreaker(
                f"{spec.provider}:{spec.model}",
                failure_threshold=cb.get("failure_threshold", 5),
                recovery_timeout_s=cb.get("recovery_timeout_s", 30),
                half_open_max_calls=cb.get("half_open_max_calls", 2),
            )
            for key, spec in self._specs.items()
        }

    def chain(self, task: Task) -> list[ModelSpec]:
        return [self._specs[k] for k in self._config["routes"][task]]

    def breaker(self, model_key: str) -> CircuitBreaker:
        return self._breakers[model_key]

    async def chat(self, task: Task, messages: list[ChatMessage]) -> RoutedResponse:
        # TODO(M2): implement retry + backoff + breaker + fallback loop described above.
        raise NotImplementedError("ModelRouter.chat — milestone M2")

    # TODO(M2): async def stream(self, task, messages) -> AsyncIterator[...]
