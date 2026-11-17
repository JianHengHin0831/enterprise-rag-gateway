from app.rag.citations import check_citations


def test_valid_citations():
    c = check_citations("Submit within 30 days [1]; the manager approves [2][1].", num_contexts=4)
    assert c.cited == (1, 2) and not c.mismatch


def test_out_of_range_citation_flagged():
    c = check_citations("Conclusion [6], see also [0].", num_contexts=4)
    assert c.invalid == (0, 6) and c.mismatch


def test_no_citation():
    c = check_citations("No relevant information was found.", num_contexts=0)
    assert not c.has_citation and not c.mismatch
