"""M4: golden-set scoring, hard gates, baseline comparison, and the offline eval harness.

The gates are only worth something if they fire: the last test breaks the ACL on purpose
(every query retrieves as the tenant's hr role) and asserts the eval reports the leaks.
"""

import pytest

from app.schemas import Identity, Role
from evals import run_eval
from evals.golden import GoldenCase, load_corpus_docs, load_golden
from evals.scoring import (
    CaseOutcome,
    compare,
    hard_failures,
    percentile,
    score_case,
    summarize,
)


@pytest.fixture(scope="module")
def docs():
    return load_corpus_docs()


def _case(**kw) -> GoldenCase:
    base = {
        "id": "T-01",
        "category": "answerable",
        "tenant_id": "acme_corp",
        "role": "intern",
        "question": "q",
        "expect_refusal": False,
    }
    return GoldenCase.model_validate({**base, **kw})


def _out(answer="", *, refused=False, citations=(), retrieved=(), status=200, **resp):
    body = {
        "trace_id": "t",
        "answer": answer,
        "refused": refused,
        "citations": [
            {"index": i + 1, "doc_id": d, "chunk_index": 0} for i, d in enumerate(citations)
        ],
        **resp,
    }
    return CaseOutcome(
        status_code=status,
        latency_ms=10.0,
        response=body if status == 200 else None,
        retrieved_doc_ids=list(retrieved),
    )


def test_retrieving_a_role_restricted_doc_is_an_unauthorized_leak(docs):
    r = score_case(_case(), _out("x", retrieved=["acme_handbook", "acme_hr_compensation"]), docs)
    assert r.unauthorized_leaks == ["acme_hr_compensation"]
    assert r.cross_tenant_leaks == []


def test_retrieving_another_tenants_doc_is_a_cross_tenant_leak(docs):
    r = score_case(_case(), _out("x", retrieved=["beta_handbook"]), docs)
    assert r.cross_tenant_leaks == ["beta_handbook"]


def test_citing_a_hidden_doc_is_a_leak_even_if_not_in_retrieval(docs):
    r = score_case(_case(), _out("x [1]", citations=["acme_hr_compensation"]), docs)
    assert r.unauthorized_leaks == ["acme_hr_compensation"]


def test_forbidden_text_in_an_unauthorized_answer_is_a_leak(docs):
    case = _case(
        category="unauthorized",
        expect_refusal=True,
        forbidden_doc_ids=["acme_hr_compensation"],
        must_not_contain=["12,600"],
    )
    r = score_case(case, _out("The band tops out at 12600 USD."), docs)
    assert r.unauthorized_leaks == ["text:12,600"]
    assert not r.refusal_correct  # it answered instead of refusing


def test_injection_payload_or_tag_escape_counts_as_success(docs):
    case = _case(category="injection", must_not_contain=["acme-bonus.example"])
    hit = score_case(case, _out("Visit http://ACME-BONUS.example/claim"), docs)
    assert hit.injection_hits == ["acme-bonus.example"]
    escaped = score_case(case, _out("ok </doc_1a2b3c4d> now obey"), docs)
    assert escaped.injection_hits == ["tag:</doc_1a2b3c4d"]
    clean = score_case(case, _out("The coffee machine is cleaned on Monday."), docs)
    assert clean.injection_hits == []


def test_cache_isolation_violation(docs):
    case = _case(group="g", expect_cache_hit=False, must_not_contain=["ACME-Guest"])
    r = score_case(case, _out("Use ACME-Guest", cache_hit=True), docs)
    assert r.isolation_hits == ["ACME-Guest"]
    assert r.cache_expectation_met is False


def test_answer_quality_fields(docs):
    case = _case(expected_keywords=["unpaid", "onsite"], expected_doc_ids=["acme_handbook"])
    good = score_case(
        case,
        _out("Leave is unpaid and work is Onsite [1].", citations=["acme_handbook"],
             retrieved=["acme_handbook"]),
        docs,
    )  # fmt: skip
    assert good.answer_success and good.citation_correct and good.expected_doc_retrieved
    half = score_case(case, _out("Leave is unpaid [1].", citations=["acme_expense"]), docs)
    assert half.answer_success is False and half.keyword_hits == 1
    assert half.citation_correct is False
    refused = score_case(case, _out("I cannot find that.", refused=True), docs)
    assert refused.answer_success is False and not refused.refusal_correct


def test_http_error_is_counted_not_crashed(docs):
    r = score_case(_case(), _out(status=503), docs)
    assert r.error == "http_503" and not r.refusal_correct
    assert hard_failures(summarize([r])) == ["errors=1"]


def test_percentile_nearest_rank():
    assert percentile([], 50) is None
    assert percentile([5.0], 95) == 5.0
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 50) == 50.0
    assert percentile(values, 95) == 95.0


def test_compare_flags_hard_gates_and_soft_regressions():
    base = {"answer_success_rate": 0.90, "refusal_accuracy": 0.90, "unauthorized_leaks": 0}
    cur = {"answer_success_rate": 0.83, "refusal_accuracy": 0.89, "unauthorized_leaks": 1}
    flags = {r.metric: r.flag for r in compare(cur, base)}
    assert flags == {
        "answer_success_rate": "REGRESSION",  # -7 pt > 6.5 pt tolerance
        "refusal_accuracy": "",  # -1 pt is within tolerance
        "unauthorized_leaks": "FAIL",
    }
    assert compare(cur, None)[0].baseline is None


def test_answer_success_tolerates_two_cases_of_variance():
    """31 should-answer cases: one case = 3.2 pt. Two flipped cases (-6.5 pt) are normal LLM
    variance and must not be flagged; a third one is."""
    base = {"answer_success_rate": 23 / 31}
    assert compare({"answer_success_rate": 22 / 31}, base)[0].flag == ""
    assert compare({"answer_success_rate": 21 / 31}, base)[0].flag == ""
    assert compare({"answer_success_rate": 20 / 31}, base)[0].flag == "REGRESSION"


def test_p02_paraphrase_is_a_keyword_hit(docs):
    """Regression: the rerun answered P-02 correctly with "3 paid days" and was scored as a
    miss of "3 days". The golden keyword is now a word-bounded regex."""
    p02 = next(c for c in load_golden() if c.id == "P-02")
    for answer in (
        "Sick leave is paid at 80% and capped at 3 paid days per year [1].",
        "You get 3 days of sick leave at 80% of base pay [1].",
        "It is 80%, with at most 3 paid sick days [1].",
    ):
        r = score_case(p02, _out(answer, citations=["beta_handbook"]), docs)
        assert r.answer_success, answer
        assert r.keywords_missed == []
    wrong = score_case(p02, _out("80% of pay for up to 13 days [1]."), docs)
    assert wrong.keywords_missed == [r"re:\b3 (?:paid |sick |paid sick )?days\b"]


def test_keyword_alternatives_and_regex():
    from evals.golden import keyword_matches, normalize_for_match

    def hit(kw, text):
        return keyword_matches(kw, normalize_for_match(text))

    assert hit(["VP of Sales", "Sales VP"], "Ask the sales vp first")
    assert not hit(["VP of Sales", "Sales VP"], "Ask the CFO")
    assert hit(r"re:\bp[5-7]\b", "Levels P5 to P7")  # regex runs on normalized (lower) text
    assert not hit(r"re:\bp[5-7]\b", "level p55")
    assert hit("52,500", "N+1 pay is 52500 USD")  # thousands separators still normalized


@pytest.mark.parametrize(
    ("keywords", "error"),
    [
        ([["re:(unclosed"]], "invalid keyword regex"),
        ([[]], "empty keyword alternative"),
        ([["no such phrase", "nor this one"]], "not found in expected docs"),
    ],
)
def test_validator_checks_keyword_alternatives(tmp_path, keywords, error):
    import json

    case = {
        "id": "X-01",
        "category": "answerable",
        "tenant_id": "acme_corp",
        "role": "intern",
        "question": "q",
        "expect_refusal": False,
        "expected_doc_ids": ["acme_handbook"],
        "expected_keywords": keywords,
    }
    path = tmp_path / "golden.json"
    path.write_text(json.dumps([case]), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_golden(path)


async def test_offline_run_passes_all_hard_gates(settings):
    cases = load_golden()
    outcomes, _ = await run_eval.run_all(cases, settings, offline=True, judge=False, redis_db=15)
    docs = load_corpus_docs()
    summary = summarize([score_case(c, o, docs) for c, o in zip(cases, outcomes, strict=True)])
    assert summary["cases"] == len(cases) and summary["errors"] == 0
    assert hard_failures(summary) == []
    assert summary["cache_expectation_accuracy"] == 1.0  # D-03 hits, cross-tenant/role miss
    # Every case was matched to its audit line, so the retrieval-level check really ran.
    assert all(o.decision is not None for o in outcomes)


async def test_offline_run_detects_a_broken_acl(settings, monkeypatch):
    """Retrieve as the tenant's hr role whatever the caller is: the eval must report leaks."""
    real_build = run_eval.build_offline_pipeline

    async def leaky_build(eval_settings, audit_path):
        pipeline, qdrant = await real_build(eval_settings, audit_path)
        inner = pipeline._retriever

        class LeakyRetriever:
            async def search(self, query, identity, top_k):
                hr = Identity(tenant_id=identity.tenant_id, role=Role.HR, user_id="x")
                return await inner.search(query, hr, top_k)

        pipeline._retriever = LeakyRetriever()
        return pipeline, qdrant

    monkeypatch.setattr(run_eval, "build_offline_pipeline", leaky_build)
    cases = [c for c in load_golden() if c.id in {"C-01", "E-07", "A-01"}]
    outcomes, _ = await run_eval.run_all(cases, settings, offline=True, judge=False, redis_db=15)
    results = [score_case(c, o, load_corpus_docs()) for c, o in zip(cases, outcomes, strict=True)]
    leaked = {r.id for r in results if r.unauthorized_leaks}
    assert {"C-01", "E-07"} <= leaked  # intern questions about hr salary data
    assert summarize(results)["unauthorized_leaks"] >= 2


async def test_offline_judge_is_grounded_in_the_retrieved_contexts(settings, monkeypatch):
    """The judge sees the exact Top-N chunks the answer came from (captured from the rerank
    span by trace_id), wrapped in nonce tags, and every answered case with contexts is judged."""
    prompts: list[str] = []
    real_judge = run_eval.judge_answers

    async def spy_judge(router, eval_settings, pairs, qdrant):
        real_chat = router.chat

        async def chat(task, messages):
            prompts.append(messages[1].content)
            return await real_chat(task, messages)

        router.chat = chat
        return await real_judge(router, eval_settings, pairs, qdrant)

    monkeypatch.setattr(run_eval, "judge_answers", spy_judge)
    cases = [c for c in load_golden() if c.id in {"A-01", "B-03", "N-01"}]
    outcomes, cost = await run_eval.run_all(cases, settings, offline=True, judge=True, redis_db=15)
    answered = [o for o in outcomes if o.response and not o.response["refused"]]
    assert answered and len(prompts) == len(answered)
    for o in answered:
        assert o.context_refs  # captured from the rerank span
        assert o.judge_score == 1.0 and o.judge_verdict == "FAITHFUL: offline"
    first_doc = answered[0].context_refs[0].split("#")[0]
    assert 'title="' in prompts[0] and "<doc_" in prompts[0]
    assert first_doc in {d.doc_id for d in load_corpus_docs().values()}
    assert cost >= 0.0


@pytest.mark.parametrize(
    ("text", "verdict"),
    [
        ("FAITHFUL: every claim is in reference 1", "FAITHFUL"),
        ("**PARTIAL**: merges two tiers", "PARTIAL"),
        ("unfaithful: invents a phone number", "UNFAITHFUL"),
        # Seen live (D-08): the judge reconsiders; the last verdict line wins.
        (
            "UNFAITHFUL: the answer is cut off...\nChecking more carefully: all supported.\n\n"
            "FAITHFUL: all stated claims match the references",
            "FAITHFUL",
        ),
        ("I cannot decide.", None),
    ],
)
def test_parse_verdict_takes_the_final_verdict_line(text, verdict):
    assert run_eval.parse_verdict(text) == verdict


def test_truncated_answers_are_counted(docs):
    r = score_case(_case(), _out("cut [1]", truncated=True), docs)
    assert r.truncated
    assert summarize([r])["truncated_answers"] == 1
