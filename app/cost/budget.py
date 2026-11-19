"""Per-tenant daily token budget in Redis: reserve -> settle.

Check-then-spend overshoots under concurrency, so:
  1. reserve(estimated) -> atomic INCRBY; if the new total exceeds the limit, DECRBY back and
     report not accepted (caller rejects with 429 or downgrades the route).
  2. settle(reservation, actual) -> INCRBY (actual - estimated) to correct the counter.
Because every in-flight request already holds its reservation, the sum of accepted
reservations never exceeds the limit; actual spend can only exceed it by the estimation error
of the requests in flight.

Key: budget:{tenant_id}:{YYYY-MM-DD} (UTC), expires after 48h.
Limits come from `budget.daily_tokens_per_tenant` in configs/models.yaml; a `default` entry
applies to tenants without their own line, and a tenant with neither is not limited (it is
still counted).
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from redis.asyncio import Redis

from app.observability.logging import log_event

logger = logging.getLogger("rag.budget")

KEY_TTL_S = 48 * 3600
OnExceeded = Literal["reject", "downgrade"]


class BudgetExceededError(Exception):
    """The tenant's daily token budget cannot cover this request (HTTP 429)."""

    def __init__(self, tenant_id: str) -> None:
        super().__init__(f"daily token budget exceeded for tenant '{tenant_id}'")
        self.tenant_id = tenant_id


def budget_key(tenant_id: str, now: datetime | None = None) -> str:
    day = (now or datetime.now(UTC)).strftime("%Y-%m-%d")
    return f"budget:{tenant_id}:{day}"


@dataclass(frozen=True)
class Reservation:
    key: str
    tokens: int
    accepted: bool


class BudgetGuard:
    def __init__(
        self,
        redis: Redis,
        daily_limits: dict[str, int],
        *,
        on_exceeded: OnExceeded = "reject",
    ) -> None:
        if on_exceeded not in ("reject", "downgrade"):
            raise ValueError(f"unsupported budget.on_exceeded '{on_exceeded}'")
        self._redis = redis
        self._limits = daily_limits
        self.on_exceeded: OnExceeded = on_exceeded

    @classmethod
    def from_config(cls, redis: Redis, config: dict[str, Any]) -> "BudgetGuard":
        b = config.get("budget", {})
        limits = {k: int(v) for k, v in (b.get("daily_tokens_per_tenant") or {}).items()}
        return cls(redis, limits, on_exceeded=b.get("on_exceeded", "reject"))

    def limit_for(self, tenant_id: str) -> int | None:
        return self._limits.get(tenant_id, self._limits.get("default"))

    async def spent(self, tenant_id: str) -> int:
        value = await self._redis.get(budget_key(tenant_id))
        return int(value or 0)

    async def reserve(self, tenant_id: str, estimated_tokens: int) -> Reservation:
        key = budget_key(tenant_id)
        tokens = max(0, int(estimated_tokens))
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.incrby(key, tokens)
            pipe.expire(key, KEY_TTL_S)
            total, _ = await pipe.execute()
        limit = self.limit_for(tenant_id)
        if limit is not None and int(total) > limit:
            await self._redis.decrby(key, tokens)
            log_event(
                logger,
                "budget_reservation_rejected",
                level=logging.WARNING,
                tenant_id=tenant_id,
                requested=tokens,
                limit=limit,
            )
            return Reservation(key=key, tokens=tokens, accepted=False)
        return Reservation(key=key, tokens=tokens, accepted=True)

    async def settle(self, reservation: Reservation, actual_tokens: int) -> None:
        """Replace the estimate by the provider-reported usage. Settling with 0 refunds the
        whole reservation (the call never reached a model)."""
        if not reservation.accepted:
            return
        delta = max(0, int(actual_tokens)) - reservation.tokens
        if delta:
            await self._redis.incrby(reservation.key, delta)
