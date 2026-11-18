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

    # Retrieval / anti-hallucination
    retrieve_top_k: int = 20
    rerank_top_n: int = 5
    rerank_threshold: float = 0.35  # TODO(M3): calibrate with golden set

    # Cache
    l1_cache_size: int = 500
    l1_cache_ttl_s: int = 60
    l2_cache_ttl_s: int = 3600

    # Audit
    audit_log_path: Path = Path("logs/audit.jsonl")


@lru_cache
def get_settings() -> Settings:
    return Settings()
