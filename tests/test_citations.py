from app.rag.citations import check_citations


def test_valid_citations():
    c = check_citations("报销需在 30 天内提交 [1]，经理审批 [2][1]。", num_contexts=4)
    assert c.cited == (1, 2) and not c.mismatch


def test_out_of_range_citation_flagged():
    c = check_citations("结论 [6]，另见 [0]。", num_contexts=4)
    assert c.invalid == (0, 6) and c.mismatch


def test_no_citation():
    c = check_citations("知识库中未找到相关依据。", num_contexts=0)
    assert not c.has_citation and not c.mismatch
