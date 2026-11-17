from pathlib import Path

from app.cost.pricing import compute_cost_usd
from app.gateway.base import Usage
from app.gateway.router import ModelSpec, load_models_config, parse_model_specs
from app.rag.ingest import load_corpus
from app.rag.prompts import load_prompt

ROOT = Path(__file__).resolve().parents[1]


def test_models_config_routes_reference_known_models():
    cfg = load_models_config(ROOT / "configs/models.yaml")
    specs = parse_model_specs(cfg)
    for task, chain in cfg["routes"].items():
        assert chain, task
        assert all(k in specs for k in chain), task
    assert cfg["routes"]["answer"][-1] == "fallback-template"


def test_compute_cost():
    spec = ModelSpec("m", "p", "m", 3.0, 15.0, 30, 1024)
    assert compute_cost_usd(Usage(prompt_tokens=1000, completion_tokens=200), spec) == 0.006
    unpriced = ModelSpec("m", "p", "m", None, None, 30, 1024)
    assert compute_cost_usd(Usage(prompt_tokens=1), unpriced) is None


def test_answer_prompt_renders_with_nonce_tag():
    p = load_prompt(ROOT / "configs/prompts", "rag_qa", 1)
    system = p.render_system(doc_tag="doc_abcd1234")
    assert "<doc_abcd1234>" in system and p.refusal_text in system
    assert "{" not in system
    assert "Question: q" in p.render_user(documents="d", query="q")


def test_mock_corpus_parses_and_flags_malicious_docs():
    chunks = load_corpus(ROOT / "data/mock_docs")
    tenants = {c.tenant_id for c in chunks}
    assert tenants == {"acme_corp", "beta_inc"}
    flagged_docs = {c.doc_id for c in chunks if c.suspicious_patterns}
    assert flagged_docs == {"acme_pantry_guide", "beta_it_faq"}  # no false positives
    # HR docs are never public
    assert all(c.allowed_roles == ["hr"] for c in chunks if c.doc_id.startswith("acme_hr_"))
