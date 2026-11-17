from app.cache.keys import build_cache_key, normalize_query, should_cache
from app.schemas import ChatResponse

BASE = dict(
    tenant_id="acme_corp",
    role="intern",
    prompt_version=1,
    rewrite_prompt_version=1,
    index_version=7,
    query="报销流程是什么？",
)


def test_normalization_merges_trivial_variants():
    assert build_cache_key(**BASE) == build_cache_key(**{**BASE, "query": "  报销流程是什么? "})
    assert normalize_query("Wi-Fi  密码！") == "wi-fi 密码"


def test_every_dimension_changes_key():
    base = build_cache_key(**BASE)
    for field, value in [
        ("tenant_id", "beta_inc"),
        ("role", "hr"),
        ("prompt_version", 2),
        ("rewrite_prompt_version", 2),
        ("index_version", 8),
        ("query", "绩效考核"),
    ]:
        assert build_cache_key(**{**BASE, field: value}) != base, field


def test_key_has_tenant_prefix():
    assert build_cache_key(**BASE).startswith("ragcache:acme_corp:")


def _resp(**kw) -> ChatResponse:
    return ChatResponse(trace_id="t", answer="a [1]", model_used="claude-sonnet", **kw)


def test_should_cache_rules():
    fallback = frozenset({"fallback-template"})
    assert should_cache(_resp(), fallback_model_keys=fallback)
    assert not should_cache(_resp(degraded=True), fallback_model_keys=fallback)
    assert not should_cache(_resp(citation_mismatch=True), fallback_model_keys=fallback)
    assert not should_cache(_resp(cache_hit=True), fallback_model_keys=fallback)
    fb = ChatResponse(trace_id="t", answer="x", model_used="fallback-template")
    assert not should_cache(fb, fallback_model_keys=fallback)
