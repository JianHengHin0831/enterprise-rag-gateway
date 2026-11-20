# Enterprise RAG Gateway

Multi-tenant, role-isolated knowledge-base QA gateway: LLM gateway (routing, circuit breaker, fallback) + ACL-filtered hybrid retrieval + anti-hallucination gates + L1/L2 cache + token budget + tracing + golden-set evals.

Plan and progress: `ROADMAP.md` (milestones, acceptance criteria) and `STATUS.md` (what is done right now). Read both before starting a task.

## Commands

Local Python is 3.11+ (Docker image uses 3.12). Windows venv paths shown; on Linux/macOS use `.venv/bin/`.

- Create env: `python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"`
- Infra only: `docker compose up -d qdrant redis`
- Full stack: `docker compose up --build`
- Run API: `.venv/Scripts/uvicorn app.main:app --reload --port 8000`
- Mint a dev JWT: `.venv/Scripts/python -m scripts.mint_token --tenant acme_corp --role hr --user u_hr_01`
- Ingest mock docs: `.venv/Scripts/python -m app.rag.ingest data/mock_docs`
- Tests: `.venv/Scripts/pytest tests/ -v`
- RAG eval: `.venv/Scripts/python -m evals.run_eval` (`--dry-run` validates the golden set only)
- Retrieval ablation (no LLM): `.venv/Scripts/python -m evals.ablation [--top-n 3]` (`--offline` = fakes)
- Load / budget race (no LLM): `.venv/Scripts/python -m scripts.loadtest --url http://localhost:8002` (start the `api-bench` launch config first)
- Lint / format: `.venv/Scripts/ruff check . && .venv/Scripts/ruff format .`

## Architecture Boundaries (CRITICAL — never violate)

1. **Identity comes only from a verified JWT.** `tenant_id`, `role`, `user_id` are produced by `app/security/auth.py::get_identity`. Never read them from request body, query params, unsigned headers, or LLM output.
2. **ACL filter is built only from `Identity`.** Use `app/security/acl.py::build_acl_filter(identity)`. Retrieval MUST pass it to Qdrant at query level — and to **every `prefetch`** in hybrid queries. Never post-filter in Python.
3. **All LLM calls go through `app/gateway/router.py`.** No `openai` / `anthropic` SDK imports or raw HTTP calls to model APIs anywhere else.
4. **Retry and fallback only before the first streamed token.** After the first token is sent, a failure ends the stream with `event: error`; never switch models mid-answer.
5. **Every cache key is built by `app/cache/keys.py::build_cache_key`**, which binds `tenant_id`, `role`, `prompt_version`, `rewrite_prompt_version`, `index_version`. Only single-turn requests (or post-rewrite queries) are cacheable. Never cache results where `degraded`, `citation_mismatch`, `truncated`, or the fallback template was used (`should_cache`).
6. **Retrieved text is untrusted.** Wrap it with `app/security/injection.py::wrap_documents` (random-nonce tags, tag-escape). Never concatenate raw chunks into prompts.
7. **No raw PII or secrets in logs.** All log fields pass through `app/observability/sanitizer.py` (the JSON log formatter does this automatically — use `log_event`, not `print`/f-string logging of payloads). Audit logs store `query_hash`, not the raw query.
8. **Budget is reserved before the LLM call** (estimated tokens) and settled with actual usage afterwards (`app/cost/budget.py`). No check-then-spend.

## Conventions

- **English only.** All code, comments, docstrings, strings, prompts, mock data, eval data and docs are written in English — not a single Chinese character anywhere in the repo. Enforced by `tests/test_english_only.py`.
- Python 3.11+ syntax, full type hints, Pydantic v2 models for all API and cross-module data (`app/schemas.py`).
- Async everywhere on the request path; one shared `httpx.AsyncClient` created in the FastAPI lifespan.
- Config: `app/config.py` (env vars prefixed `RAG_`), model/routing in `configs/models.yaml`, prompts versioned in `configs/prompts/<name>_v<N>.yaml`. Changing prompt text = new version file, never edit an old one.
- Unimplemented parts are marked `TODO(M1)`…`TODO(M4)` matching `ROADMAP.md`. Acceptance tests for future milestones exist as `pytest.mark.skip(reason="M2: ...")` — remove the skip when implementing, do not rewrite the assertion to make it pass.
- Every invariant above has a unit test; adding a new invariant means adding a test.
- Tests must not hit the network or real LLMs: use `app/gateway/providers/fake.py::FakeLLM` and fakes for Qdrant/Redis.
- Work one milestone at a time; when a milestone's acceptance passes, update `STATUS.md` and `ROADMAP.md` checkboxes.

## Never

- Edit or read `.env` (secrets). Use `.env.example` for documenting new variables.
- Commit real API keys, real personal data, or regenerate mock PII with realistic-looking real people.
- Lower the rerank threshold or loosen ACL tests to make an eval pass — report the regression instead.
