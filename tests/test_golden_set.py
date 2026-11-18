import json

import pytest

from evals.run_eval import Category, load_golden, normalize_for_match


def test_golden_set_valid():
    cases = load_golden()
    assert {c.category for c in cases} == set(Category)
    assert all(c.expect_refusal for c in cases if c.category is Category.NO_ANSWER)
    assert 30 <= len(cases) <= 60


def test_normalize_for_match():
    assert normalize_for_match("Pay 52,500 USD") == "pay 52500 usd"
    assert normalize_for_match("1,000,000") == "1000000"
    assert normalize_for_match("a,b and 1, 2") == "a,b and 1, 2"


def _write(tmp_path, case: dict):
    base = {
        "id": "X-01",
        "category": "answerable",
        "tenant_id": "acme_corp",
        "role": "intern",
        "question": "q",
        "expect_refusal": False,
    }
    path = tmp_path / "golden.json"
    path.write_text(json.dumps([{**base, **case}]), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("case", "error"),
    [
        (
            {"category": "unauthorized", "forbidden_doc_ids": ["acme_handbook"]},
            "actually visible",
        ),
        ({"expected_doc_ids": ["acme_hr_compensation"]}, "not visible"),
        ({"expected_doc_ids": ["beta_handbook"]}, "not visible"),
        ({"expected_doc_ids": ["no_such_doc"]}, "unknown doc_id"),
        (
            {"expected_doc_ids": ["acme_handbook"], "expected_keywords": ["99999 bonus"]},
            "not found in expected docs",
        ),
        ({"group": "g", "expect_cache_hit": True}, "no earlier identical case"),
        ({"expect_refusal": True, "expected_keywords": ["x"]}, "cannot have expected_keywords"),
    ],
)
def test_validator_rejects_bad_cases(tmp_path, case, error):
    with pytest.raises(ValueError, match=error):
        load_golden(_write(tmp_path, case))
