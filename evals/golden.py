"""Golden-set schema, loader and validator (`evals/golden_set.json`).

Matching uses `normalize_for_match` on both sides (case-insensitive, thousands separators
removed), so "52,500" matches "52500". `expect_cache_hit` (null = not checked) only makes
sense inside a `group` of cases that repeat the same question; cases run in file order.
"""

import json
import re
from collections import Counter
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from app.cache.keys import normalize_query
from app.rag.ingest import SourceDoc, parse_doc
from app.schemas import PUBLIC_ACCESS, Role

GOLDEN_PATH = Path(__file__).parent / "golden_set.json"
CORPUS_ROOT = Path(__file__).resolve().parents[1] / "data" / "mock_docs"
# Tags whose expected keywords are computed results, so they need not appear in the docs.
UNGROUNDED_TAGS = frozenset({"calculation"})

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_SPACES = re.compile(r"\s+")


def normalize_for_match(text: str) -> str:
    return _SPACES.sub(" ", _THOUSANDS.sub("", text.lower()))


# An expected keyword is a string or a list of alternatives (any one counts as a hit), so a
# paraphrase such as "3 paid days" for "3 days" is not scored as a miss. An alternative
# starting with `re:` is a regular expression searched in the normalized (lowercased,
# thousands-separator-free) text, e.g. "re:\b3 (?:paid )?days".
REGEX_PREFIX = "re:"
Keyword = str | list[str]


def keyword_alternatives(keyword: Keyword) -> list[str]:
    return [keyword] if isinstance(keyword, str) else list(keyword)


def keyword_label(keyword: Keyword) -> str:
    return " | ".join(keyword_alternatives(keyword))


def keyword_matches(keyword: Keyword, text_norm: str) -> bool:
    """True when any alternative occurs in `text_norm` (already `normalize_for_match`ed)."""
    for alt in keyword_alternatives(keyword):
        if alt.startswith(REGEX_PREFIX):
            try:
                if re.search(alt[len(REGEX_PREFIX) :], text_norm):
                    return True
            except re.error:
                continue  # reported by the golden-set validator
        elif normalize_for_match(alt) in text_norm:
            return True
    return False


TARGET_MIX = {
    "answerable": 0.50,
    "no_answer": 0.20,
    "unauthorized+cross_tenant": 0.15,
    "injection": 0.15,
}


class Category(StrEnum):
    ANSWERABLE = "answerable"
    NO_ANSWER = "no_answer"
    UNAUTHORIZED = "unauthorized"
    CROSS_TENANT = "cross_tenant"
    INJECTION = "injection"


class GoldenCase(BaseModel):
    id: str
    category: Category
    tenant_id: str
    role: Role
    question: str
    expected_keywords: list[Keyword] = Field(default_factory=list)
    expected_doc_ids: list[str] = Field(default_factory=list)
    forbidden_doc_ids: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    expect_refusal: bool
    tags: list[str] = Field(default_factory=list)
    group: str | None = None
    expect_cache_hit: bool | None = None
    reference_answer: str | None = None
    notes: str | None = None


def load_corpus_docs(root: Path = CORPUS_ROOT) -> dict[str, SourceDoc]:
    return {doc.doc_id: doc for doc in map(parse_doc, sorted(root.rglob("*.md")))}


def can_access(doc: SourceDoc, tenant_id: str, role: Role) -> bool:
    return doc.tenant_id == tenant_id and bool({role.value, PUBLIC_ACCESS} & set(doc.allowed_roles))


def _check_case(c: GoldenCase, docs: dict[str, SourceDoc]) -> list[str]:
    errors = []
    if c.category in (Category.UNAUTHORIZED, Category.CROSS_TENANT) and not c.forbidden_doc_ids:
        errors.append(f"{c.category} case needs forbidden_doc_ids")
    if c.expect_refusal and c.expected_keywords:
        errors.append("a refusal case cannot have expected_keywords")
    for kw in c.expected_keywords:
        alts = keyword_alternatives(kw)
        if not alts or not all(alts):
            errors.append(f"empty keyword alternative in {kw!r}")
        for alt in alts:
            if alt.startswith(REGEX_PREFIX):
                try:
                    re.compile(alt[len(REGEX_PREFIX) :])
                except re.error as exc:
                    errors.append(f"invalid keyword regex {alt!r}: {exc}")
    if c.expect_cache_hit is not None and c.group is None:
        errors.append("expect_cache_hit requires a group")
    for doc_id in c.expected_doc_ids + c.forbidden_doc_ids:
        if doc_id not in docs:
            errors.append(f"unknown doc_id {doc_id}")
    for doc_id in c.expected_doc_ids:
        if doc_id in docs and not can_access(docs[doc_id], c.tenant_id, c.role):
            errors.append(f"expected doc {doc_id} is not visible to {c.tenant_id}/{c.role}")
    for doc_id in c.forbidden_doc_ids:
        if doc_id in docs and can_access(docs[doc_id], c.tenant_id, c.role):
            errors.append(f"forbidden doc {doc_id} is actually visible to {c.tenant_id}/{c.role}")
    if not UNGROUNDED_TAGS & set(c.tags):
        source = normalize_for_match(
            " ".join(docs[d].body for d in c.expected_doc_ids if d in docs)
        )
        for kw in c.expected_keywords:
            if not keyword_matches(kw, source):
                errors.append(f"keyword {keyword_label(kw)!r} not found in expected docs")
    return errors


def _check_cache_groups(cases: list[GoldenCase]) -> list[str]:
    errors = []
    seen: set[tuple[str, str, str, str]] = set()
    for c in cases:
        if c.group is None:
            continue
        key = (c.group, c.tenant_id, c.role.value, normalize_query(c.question))
        if c.expect_cache_hit is True and key not in seen:
            errors.append(f"{c.id}: expects a cache hit but no earlier identical case in group")
        if c.expect_cache_hit is False and key in seen:
            errors.append(f"{c.id}: expects a miss but repeats an earlier case in group")
        seen.add(key)
    return errors


def load_golden(path: Path = GOLDEN_PATH, corpus_root: Path = CORPUS_ROOT) -> list[GoldenCase]:
    cases = [GoldenCase.model_validate(c) for c in json.loads(path.read_text(encoding="utf-8"))]
    ids = [c.id for c in cases]
    dupes = [i for i, n in Counter(ids).items() if n > 1]
    if dupes:
        raise ValueError(f"duplicate ids: {dupes}")
    docs = load_corpus_docs(corpus_root)
    errors = [f"{c.id}: {e}" for c in cases for e in _check_case(c, docs)]
    errors += _check_cache_groups(cases)
    if errors:
        raise ValueError("\n  " + "\n  ".join(errors))
    return cases


def category_mix(cases: list[GoldenCase]) -> dict[str, float]:
    counts = Counter(c.category.value for c in cases)
    n = len(cases) or 1
    return {
        "answerable": counts["answerable"] / n,
        "no_answer": counts["no_answer"] / n,
        "unauthorized+cross_tenant": (counts["unauthorized"] + counts["cross_tenant"]) / n,
        "injection": counts["injection"] / n,
    }
