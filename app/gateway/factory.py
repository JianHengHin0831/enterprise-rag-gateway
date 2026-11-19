"""Build provider adapters from configs/models.yaml + Settings (API keys, base_url overrides).

A provider is registered only when it can actually serve requests: its adapter is
implemented and, for remote providers, its API key is set. The router then skips models of
unregistered providers (`ModelRouter.available_chain`).
"""

import logging
from typing import Any

import httpx

from app.config import Settings
from app.gateway.base import BaseLLMProvider
from app.gateway.providers.openai_compat import OpenAICompatProvider
from app.gateway.providers.template import TemplateProvider
from app.observability.logging import log_event

logger = logging.getLogger("rag.gateway")

_IMPLEMENTED_REMOTE = {"openai_compat"}  # TODO(M2): add "anthropic"


def api_key_for(settings: Settings, env_name: str) -> str | None:
    """`RAG_DASHSCOPE_API_KEY` -> `settings.dashscope_api_key`."""
    field = env_name.lower().removeprefix("rag_")
    secret = getattr(settings, field, None)
    if secret is None:
        return None
    value = secret.get_secret_value().strip()
    return value or None


def build_providers(
    config: dict[str, Any], settings: Settings, http: httpx.AsyncClient
) -> dict[str, BaseLLMProvider]:
    providers: dict[str, BaseLLMProvider] = {}
    skipped: dict[str, str] = {}
    for name, spec in config.get("providers", {}).items():
        kind = spec.get("type")
        if kind == "template":
            providers[name] = TemplateProvider()
            continue
        if kind not in _IMPLEMENTED_REMOTE:
            skipped[name] = "adapter_not_implemented"
            continue
        key = api_key_for(settings, spec.get("api_key_env", ""))
        if key is None:
            skipped[name] = "no_api_key"
            continue
        providers[name] = OpenAICompatProvider(
            name=name,
            base_url=settings.provider_base_urls.get(name, spec["base_url"]),
            api_key=key,
            client=http,
        )
    log_event(logger, "providers_configured", enabled=sorted(providers), skipped=skipped)
    return providers
