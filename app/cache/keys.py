"""Tenant/role-isolated cache keys and cacheability rules.

Every component that changes the answer must be in the key:
  tenant_id, role          -> no cross-tenant / cross-role leakage
  prompt_version           -> answer prompt change invalidates
  rewrite_prompt_version   -> rewrite prompt change alters retrieval
  index_version            -> bump per tenant on re-ingest; old entries die naturally
  query                    -> the REWRITTEN (self-contained) query, never a raw multi-turn one
"""

import hashlib
import json
import re
import unicodedata

from app.schemas import ChatResponse

_WS = re.compile(r"\s+")
_TRAILING_PUNCT = re.compile(r"[\s?!.,]+$")


def normalize_query(query: str) -> str:
    q = unicodedata.normalize("NFKC", query).strip().lower()
    q = _WS.sub(" ", q)
    return _TRAILING_PUNCT.sub("", q)


def index_version_key(tenant_id: str) -> str:
    """Redis key holding a tenant's current index_version (written by ingest, read by cache)."""
    return f"index_version:{tenant_id}"


def build_cache_key(
    *,
    tenant_id: str,
    role: str,
    prompt_version: int,
    rewrite_prompt_version: int,
    index_version: int,
    query: str,
) -> str:
    material = json.dumps(
        [
            tenant_id,
            role,
            prompt_version,
            rewrite_prompt_version,
            index_version,
            normalize_query(query),
        ],
        ensure_ascii=False,
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    # Tenant prefix allows SCAN-based ops/debugging per tenant; the digest guarantees isolation.
    return f"ragcache:{tenant_id}:{digest}"


def should_cache(response: ChatResponse, *, fallback_model_keys: frozenset[str]) -> bool:
    if response.degraded or response.citation_mismatch or response.truncated:
        return False
    if response.model_used in fallback_model_keys:
        return False
    return not response.cache_hit
