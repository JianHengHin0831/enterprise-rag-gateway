"""Golden-set evaluation.

python -m evals.run_eval --dry-run                          # validate the golden set only
python -m evals.run_eval --compare evals/baselines/latest.json
python -m evals.run_eval --save-baseline                    # only when explicitly intended

Scoring semantics (M4):
- Hard gates (any failure is a regression): `forbidden_doc_ids` must never appear in the
  request's retrieved_doc_ids, and `must_not_contain` must never appear in the answer.
- Soft metrics: `expected_keywords` hit rate, refusal correctness (`expect_refusal`),
  expected-doc recall, citation mismatch, and LLM-as-judge against `reference_answer`.
- Matching uses `normalize_for_match` on both sides (case-insensitive, thousands separators
  removed), so "52,500" matches "52500".
- Cases run in file order with an empty cache. `expect_cache_hit` (null = not checked) only
  makes sense inside a `group` of cases that repeat the same question.
"""

import argparse
import json
import re
import sys
from collections import Counter
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

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
    expected_keywords: list[str] = Field(default_factory=list)
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
            if normalize_for_match(kw) not in source:
                errors.append(f"keyword {kw!r} not found in expected docs")
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--save-baseline", action="store_true")
    args = parser.parse_args()

    try:
        cases = load_golden()
    except (ValidationError, ValueError) as exc:
        print(f"golden set invalid: {exc}")
        return 1

    print(f"{len(cases)} cases")
    for k, v in category_mix(cases).items():
        print(f"  {k:<28} {v:6.1%}  (target {TARGET_MIX[k]:.0%})")
    tag_counts = Counter(t for c in cases for t in c.tags)
    print("  tags: " + ", ".join(f"{t}={n}" for t, n in sorted(tag_counts.items())))

    if args.dry_run:
        return 0

    # TODO(M4): for each case, mint a token for (tenant_id, role), call the pipeline in-process,
    # then score: keyword hit, refusal correctness, forbidden_doc_ids absent from
    # retrieved_doc_ids (retrieval-level), must_not_contain absent from answer, citation
    # mismatch, latency, cost. Compare with --compare baseline; write evals/reports/<ts>.json.
    print("full eval not implemented yet (M4)")
    return 2


if __name__ == "__main__":
    sys.exit(main())
