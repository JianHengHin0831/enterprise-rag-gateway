"""L1 (in-process TTLCache) + L2 (Redis) answer cache.

- get(key): L1 -> L2 (an L2 hit backfills L1) -> None. A hit is returned as a copy marked
  `cache_hit=True`, `cost_usd=0`, with the current request's trace_id.
- set(key, ChatResponse): write both; L2 expires after `l2_ttl_s`.
- index_version(tenant): `index_version:{tenant_id}` in Redis (written by ingest), cached in
  a small L1 for `index_version_ttl_s` so a re-ingest takes effect within seconds without a
  Redis round-trip per request.

The cache is an optimisation: a Redis failure on get/set is logged and treated as a miss.
Cache keys come only from `app/cache/keys.py::build_cache_key`.
"""

import logging

from cachetools import TTLCache
from pydantic import ValidationError
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.cache.keys import index_version_key
from app.observability.logging import log_event
from app.observability.trace import current_trace_id
from app.schemas import ChatResponse

logger = logging.getLogger("rag.cache")


class LayeredCache:
    def __init__(
        self,
        redis: Redis,
        *,
        l1_size: int,
        l1_ttl_s: int,
        l2_ttl_s: int,
        index_version_ttl_s: float = 5.0,
    ) -> None:
        self._redis = redis
        self._l1: TTLCache[str, ChatResponse] = TTLCache(maxsize=l1_size, ttl=l1_ttl_s)
        self._l2_ttl_s = l2_ttl_s
        self._versions: TTLCache[str, int] = TTLCache(maxsize=1024, ttl=index_version_ttl_s)

    async def get(self, key: str) -> ChatResponse | None:
        value = self._l1.get(key)
        layer = "l1"
        if value is None:
            layer = "l2"
            try:
                raw = await self._redis.get(key)
            except RedisError as exc:
                log_event(logger, "cache_error", level=logging.WARNING, op="get", error=str(exc))
                return None
            if raw is None:
                return None
            try:
                value = ChatResponse.model_validate_json(raw)
            except ValidationError:
                log_event(logger, "cache_corrupt_entry", level=logging.WARNING)
                return None
            self._l1[key] = value
        log_event(logger, "cache_hit", layer=layer)
        return value.model_copy(
            update={"cache_hit": True, "cost_usd": 0.0, "trace_id": current_trace_id() or ""}
        )

    async def set(self, key: str, value: ChatResponse) -> None:
        self._l1[key] = value
        try:
            await self._redis.set(key, value.model_dump_json(), ex=self._l2_ttl_s)
        except RedisError as exc:
            log_event(logger, "cache_error", level=logging.WARNING, op="set", error=str(exc))

    async def index_version(self, tenant_id: str) -> int | None:
        """None when Redis cannot be read: the caller then skips the cache for this request
        (an unknown version must never be treated as a match)."""
        cached = self._versions.get(tenant_id)
        if cached is not None:
            return cached
        try:
            raw = await self._redis.get(index_version_key(tenant_id))
        except RedisError as exc:
            log_event(logger, "cache_error", level=logging.WARNING, op="version", error=str(exc))
            return None
        version = int(raw or 0)
        self._versions[tenant_id] = version
        return version
