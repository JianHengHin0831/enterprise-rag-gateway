"""L1 (in-process TTLCache) + L2 (Redis) answer cache.

TODO(M2):
- get(key): L1 -> L2 (on L2 hit, backfill L1) -> None
- set(key, ChatResponse): write both; L2 with `l2_cache_ttl_s`
- index_version lookup: `index_version:{tenant_id}` in Redis, cached in L1 for a few seconds
  so a re-ingest takes effect quickly without a Redis round-trip per request.
"""

from cachetools import TTLCache
from redis.asyncio import Redis

from app.schemas import ChatResponse


class LayeredCache:
    def __init__(self, redis: Redis, *, l1_size: int, l1_ttl_s: int, l2_ttl_s: int) -> None:
        self._redis = redis
        self._l1: TTLCache[str, ChatResponse] = TTLCache(maxsize=l1_size, ttl=l1_ttl_s)
        self._l2_ttl_s = l2_ttl_s

    async def get(self, key: str) -> ChatResponse | None:
        raise NotImplementedError("LayeredCache.get — milestone M2")

    async def set(self, key: str, value: ChatResponse) -> None:
        raise NotImplementedError("LayeredCache.set — milestone M2")

    async def index_version(self, tenant_id: str) -> int:
        raise NotImplementedError("LayeredCache.index_version — milestone M2")
