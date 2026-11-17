"""Golden-set evaluation.

python -m evals.run_eval --dry-run                          # validate the golden set only
python -m evals.run_eval --compare evals/baselines/latest.json
python -m evals.run_eval --save-baseline                    # only when explicitly intended
"""

import argparse
import json
import sys
from collections import Counter
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from app.schemas import Role

GOLDEN_PATH = Path(__file__).parent / "golden_set.json"
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
    notes: str | None = None


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenCase]:
    cases = [GoldenCase.model_validate(c) for c in json.loads(path.read_text(encoding="utf-8"))]
    ids = [c.id for c in cases]
    dupes = [i for i, n in Counter(ids).items() if n > 1]
    if dupes:
        raise ValueError(f"duplicate ids: {dupes}")
    for c in cases:
        if c.category in (Category.UNAUTHORIZED, Category.CROSS_TENANT) and not c.forbidden_doc_ids:
            raise ValueError(f"{c.id}: {c.category} case needs forbidden_doc_ids")
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
