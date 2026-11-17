"""Three-state circuit breaker, one instance per `provider:model`.

CLOSED --(failure_threshold consecutive failures)--> OPEN
OPEN   --(recovery_timeout_s elapsed)-------------> HALF_OPEN
HALF_OPEN: at most `half_open_max_calls` probes in flight;
           a probe success -> CLOSED, a probe failure -> OPEN (timer restarts).

Not thread-safe by design: used from a single asyncio event loop and no method awaits, so
state transitions are atomic with respect to other coroutines. State is per process; with
multiple uvicorn workers each worker trips independently (documented trade-off).
"""

import time
from collections.abc import Callable
from enum import StrEnum


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"circuit open: {name}")
        self.name = name


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 30.0,
        half_open_max_calls: int = 2,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout_s = recovery_timeout_s
        self._half_open_max_calls = half_open_max_calls
        self._clock = clock

        self._state = BreakerState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._half_open_in_flight = 0

    @property
    def state(self) -> BreakerState:
        if (
            self._state is BreakerState.OPEN
            and self._clock() - self._opened_at >= self._recovery_timeout_s
        ):
            self._state = BreakerState.HALF_OPEN
            self._half_open_in_flight = 0
        return self._state

    def allow_request(self) -> bool:
        """Call before each attempt. In HALF_OPEN this reserves a probe slot."""
        state = self.state
        if state is BreakerState.CLOSED:
            return True
        if state is BreakerState.OPEN:
            return False
        if self._half_open_in_flight < self._half_open_max_calls:
            self._half_open_in_flight += 1
            return True
        return False

    def record_success(self) -> None:
        self._state = BreakerState.CLOSED
        self._consecutive_failures = 0
        self._half_open_in_flight = 0

    def record_failure(self) -> None:
        """Record a RETRYABLE failure only (timeouts, 429, 5xx)."""
        if self.state is BreakerState.HALF_OPEN:
            self._trip()
            return
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold:
            self._trip()

    def _trip(self) -> None:
        self._state = BreakerState.OPEN
        self._opened_at = self._clock()
        self._half_open_in_flight = 0
