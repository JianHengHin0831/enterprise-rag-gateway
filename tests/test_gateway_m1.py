"""Provider registration from models.yaml + the M1 single-model router path."""

import httpx
import pytest
from pydantic import SecretStr

from app.gateway.base import ChatMessage, NonRetryableLLMError, RetryableLLMError
from app.gateway.factory import api_key_for, build_providers
from app.gateway.providers.fake import FakeLLM
from app.gateway.providers.openai_compat import OpenAICompatProvider
from app.gateway.providers.template import FALLBACK_TEXT, TemplateProvider
from app.gateway.router import ModelRouter, load_models_config

MSGS = [ChatMessage(role="user", content="hi")]


@pytest.fixture
def models_config(settings):
    return load_models_config(settings.models_config_path)


def test_api_key_lookup(settings):
    s = settings.model_copy(update={"dashscope_api_key": SecretStr(" k ")})
    assert api_key_for(s, "RAG_DASHSCOPE_API_KEY") == "k"
    assert api_key_for(s, "RAG_ANTHROPIC_API_KEY") is None
    assert api_key_for(s, "RAG_NOT_A_FIELD") is None
    blank = settings.model_copy(update={"dashscope_api_key": SecretStr("  ")})
    assert api_key_for(blank, "RAG_DASHSCOPE_API_KEY") is None


async def test_build_providers_only_registers_usable_ones(settings, models_config):
    s = settings.model_copy(
        update={
            "dashscope_api_key": SecretStr("k"),
            "anthropic_api_key": SecretStr("k"),
            "provider_base_urls": {"dashscope": "https://intl.example/v1"},
        }
    )
    async with httpx.AsyncClient() as http:
        providers = build_providers(models_config, s, http)
    assert set(providers) == {"dashscope", "local"}  # anthropic adapter is M2
    assert isinstance(providers["dashscope"], OpenAICompatProvider)
    assert providers["dashscope"]._base_url == "https://intl.example/v1"
    assert isinstance(providers["local"], TemplateProvider)


async def test_without_keys_only_template_is_registered(settings, models_config):
    async with httpx.AsyncClient() as http:
        assert set(build_providers(models_config, settings, http)) == {"local"}


async def test_router_uses_first_configured_model_of_chain(models_config):
    llm = FakeLLM("dashscope", default="Answer [1]")
    router = ModelRouter(
        config=models_config, providers={"dashscope": llm, "local": TemplateProvider()}
    )
    assert [m.key for m in router.available_chain("answer")] == [
        "qwen-plus",
        "fallback-template",
    ]
    result = await router.chat("answer", MSGS)
    assert result.model_key == "qwen-plus" and not result.degraded
    assert result.response.text == "Answer [1]"
    assert llm.calls == ["qwen-plus"]


async def test_router_template_only_is_degraded(models_config):
    router = ModelRouter(config=models_config, providers={"local": TemplateProvider()})
    result = await router.chat("answer", MSGS)
    assert result.model_key == "fallback-template"
    assert result.degraded and result.response.text == FALLBACK_TEXT


async def test_router_without_any_model_raises(models_config):
    router = ModelRouter(config=models_config, providers={})
    with pytest.raises(NonRetryableLLMError, match="no configured model"):
        await router.chat("answer", MSGS)


async def test_router_m1_propagates_provider_errors(models_config):
    llm = FakeLLM("dashscope", script=[RetryableLLMError("503", status_code=503)])
    router = ModelRouter(
        config=models_config, providers={"dashscope": llm, "local": TemplateProvider()}
    )
    with pytest.raises(RetryableLLMError):  # TODO(M2): falls back instead
        await router.chat("answer", MSGS)
