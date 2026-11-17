from evals.run_eval import Category, load_golden


def test_golden_set_valid():
    cases = load_golden()
    assert {c.category for c in cases} == set(Category)
    assert all(c.expect_refusal for c in cases if c.category is Category.NO_ANSWER)
