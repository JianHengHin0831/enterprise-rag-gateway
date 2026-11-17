from app.gateway.circuit_breaker import BreakerState, CircuitBreaker


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def make(clock: Clock) -> CircuitBreaker:
    return CircuitBreaker(
        "p:m", failure_threshold=5, recovery_timeout_s=30, half_open_max_calls=2, clock=clock
    )


def test_opens_after_threshold():
    cb = make(Clock())
    for _ in range(4):
        cb.record_failure()
    assert cb.state is BreakerState.CLOSED
    cb.record_failure()
    assert cb.state is BreakerState.OPEN
    assert not cb.allow_request()


def test_success_resets_consecutive_count():
    cb = make(Clock())
    for _ in range(4):
        cb.record_failure()
    cb.record_success()
    for _ in range(4):
        cb.record_failure()
    assert cb.state is BreakerState.CLOSED


def test_half_open_after_cooldown_limits_probes():
    clock = Clock()
    cb = make(clock)
    for _ in range(5):
        cb.record_failure()
    clock.t = 29.9
    assert cb.state is BreakerState.OPEN
    clock.t = 30.0
    assert cb.state is BreakerState.HALF_OPEN
    assert cb.allow_request() and cb.allow_request()
    assert not cb.allow_request()  # third probe blocked


def test_half_open_success_closes():
    clock = Clock()
    cb = make(clock)
    for _ in range(5):
        cb.record_failure()
    clock.t = 31
    assert cb.allow_request()
    cb.record_success()
    assert cb.state is BreakerState.CLOSED and cb.allow_request()


def test_half_open_failure_reopens_and_restarts_timer():
    clock = Clock()
    cb = make(clock)
    for _ in range(5):
        cb.record_failure()
    clock.t = 31
    assert cb.allow_request()
    cb.record_failure()
    assert cb.state is BreakerState.OPEN
    clock.t = 60
    assert cb.state is BreakerState.OPEN
    clock.t = 61
    assert cb.state is BreakerState.HALF_OPEN
