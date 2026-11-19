from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    env: str = "dev"

    # Auth
    jwt_secret: SecretStr = SecretStr("dev-only-insecure-secret-change-me")
    jwt_algorithm: str = "HS256"
    known_tenants: frozenset[str] = frozenset({"acme_corp", "beta_inc"})

    # Infra
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "kb_chunks"
    redis_url: str = "redis://localhost:6379/0"

    # LLM providers. configs/models.yaml names each key by its env var (`api_key_env`);
    # the matching field here is that name without the RAG_ prefix, lowercased.
    anthropic_api_key: SecretStr | None = None
    dashscope_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    # Optional per-provider base_url override, e.g.
    # RAG_PROVIDER_BASE_URLS='{"dashscope": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"}'
    provider_base_urls: dict[str, str] = {}

    # Embeddings (app/rag/embeddings.py)
    embedder: Literal["bge-m3", "fake"] = "bge-m3"
    embed_model_name: str = "BAAI/bge-m3"
    embed_device: str | None = None  # e.g. "cpu", "cuda:0"; None = FlagEmbedding default
    embed_use_fp16: bool = False  # True only on GPU
    embed_batch_size: int = 16
    embed_max_length: int = 512
    fake_embed_dim: int = 64

    # Config files
    models_config_path: Path = Path("configs/models.yaml")
    prompts_dir: Path = Path("configs/prompts")
    answer_prompt_version: int = 1
    rewrite_prompt_version: int = 1

    # Retrieval / anti-hallucination. The threshold is in the reranker's native score scale
    # and only valid for the reranker it was calibrated on (`python -m evals.sweep_threshold`).
    retrieve_top_k: int = 20
    reranker: Literal["bge-reranker", "fake"] = "bge-reranker"
    rerank_model_name: str = "BAAI/bge-reranker-v2-m3"
    # Reranker device / precision; None = follow embed_device / embed_use_fp16. Lets the
    # reranker sit on a small GPU while the embedder stays on CPU. fp16 shifts scores slightly,
    # so re-run the threshold sweep before enabling it.
    rerank_device: str | None = None
    rerank_use_fp16: bool | None = None
    # Contexts sent to the LLM. 8, not 5: in eval run 4, four multi-part questions (A-06/07/08,
    # D-08) missed a fact whose chunk ranked just outside the Top-5. The gate uses only the
    # top-1 score, so the calibrated threshold is unaffected.
    rerank_top_n: int = 8
    rerank_threshold: float = 0.03  # bge-reranker-v2-m3, golden-set sweep 2026-11-19

    # Cache
    l1_cache_size: int = 500
    l1_cache_ttl_s: int = 60
    l2_cache_ttl_s: int = 3600
    index_version_cache_ttl_s: float = 5.0  # how fast a re-ingest invalidates cached answers

    # Query rewrite (route `rewrite`): multi-turn requests and queries shorter than
    # `rewrite_min_words` words; on error or after `rewrite_timeout_s` the raw query is used.
    rewrite_min_words: int = 4
    rewrite_timeout_s: float = 6.0  # rewrite via OpenRouter takes ~3 s; 3.0 timed out

    # Audit (append-only JSONL, one line per /v1/chat request)
    audit_log_path: Path = Path("logs/audit.jsonl")

    # Optional LangFuse trace export (OTLP/HTTP); off unless both keys are set.
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
