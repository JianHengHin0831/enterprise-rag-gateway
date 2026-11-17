"""Per-tenant daily token budget in Redis: reserve -> settle.

Check-then-spend overshoots under concurrency, so:
  1. reserve(estimated) -> atomic INCRBY; if the new total exceeds the limit, DECRBY back and
     report exceeded (caller rejects with 429 or downgrades the route).
  2. settle(reservation, actual) -> INCRBY (actual - estimated) to correct the counter.
Key: budget:{tenant_id}:{YYYY-MM-DD} (UTC), expires after 48h.
"""

from dataclasses import dataclass
from datetime import UTC, datetime

from redis.asyncio import Redis


def budget_key(tenant_id: str, now: datetime | None = None) -> str:
    day = (now or datetime.now(UTC)).strftime("%Y-%m-%d")
    return f"budget:{tenant_id}:{day}"


@dataclass(frozen=True)
class Reservation:
    key: str
    tokens: int
    accepted: bool


class BudgetGuard:
    def __init__(self, redis: Redis, daily_limits: dict[str, int]) -> None:
        self._redis = redis
        self._limits = daily_limits

    async def reserve(self, tenant_id: str, estimated_tokens: int) -> Reservation:
        raise NotImplementedError("BudgetGuard.reserve — milestone M2")  # TODO(M2)

    async def settle(self, reservation: Reservation, actual_tokens: int) -> None:
        raise NotImplementedError("BudgetGuard.settle — milestone M2")  # TODO(M2)
