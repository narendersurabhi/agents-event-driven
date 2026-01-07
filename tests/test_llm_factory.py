import os
import types

import pytest

from core.llm_factory import get_async_llm_client, get_sync_llm_client
from core.llm_client import (
    AsyncOpenAILLMClient,
    OpenAILLMClient,
    AsyncOpenAIGPT5LLMClient,
    OpenAIGPT5LLMClient,
    AsyncClaudeLLMClient,
    ClaudeLLMClient,
    AsyncGeminiLLMClient,
    GeminiLLMClient,
)


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    from core import config as cfg

    # Prevent tests from accidentally reading your real .env file.
    monkeypatch.setenv("DOTENV_PATH", "tests/.env.DO_NOT_USE")
    # Factory requires timeout to be configured (used by all providers).
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "60")
    cfg._config_adapter.cache_clear()  # type: ignore[attr-defined]

    for key in ["LLM_PROVIDER"]:
        monkeypatch.delenv(key, raising=False)
    yield
    cfg._config_adapter.cache_clear()  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "provider,expected_async,expected_sync",
    [
        ("openai", AsyncOpenAILLMClient, OpenAILLMClient),
        ("openai-gpt5", AsyncOpenAIGPT5LLMClient, OpenAIGPT5LLMClient),
        ("claude", AsyncClaudeLLMClient, ClaudeLLMClient),
        ("gemini", AsyncGeminiLLMClient, GeminiLLMClient),
    ],
)
def test_factory_returns_expected_clients(monkeypatch, provider, expected_async, expected_sync):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    a = get_async_llm_client()
    s = get_sync_llm_client()
    assert isinstance(a, expected_async)
    assert isinstance(s, expected_sync)


def test_factory_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "does-not-exist")
    with pytest.raises(ValueError):
        get_async_llm_client()
    with pytest.raises(ValueError):
        get_sync_llm_client()


def test_gpt5_temperature_stripped(monkeypatch):
    calls = {}

    class FakeResp:
        def __init__(self):
            self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]
            self.usage = None

    class FakeChat:
        def __init__(self):
            self.completions = self
        def create(self, **kwargs):
            calls.update(kwargs)
            return FakeResp()

    class FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.chat = FakeChat()

    monkeypatch.setenv("LLM_PROVIDER", "openai-gpt5")
    monkeypatch.setenv("OPENAI_MAX_COMPLETION_TOKENS", "0")
    monkeypatch.setenv("OPENAI_MAX_OUTPUT_TOKENS", "0")
    monkeypatch.setattr("core.llm_client.OpenAI", FakeOpenAI)

    llm = get_sync_llm_client()
    out = llm.chat(messages=[{"role": "user", "content": "hi"}], model="gpt-5", temperature=0.3)
    assert out == "ok"
    assert calls.get("temperature") is None  # stripped for gpt-5


def test_claude_client(monkeypatch):
    class FakeResp:
        def __init__(self):
            self.content = [types.SimpleNamespace(text="hello")]  # anthropic message format
            self.usage = None

    class FakeMessages:
        def create(self, **kwargs):
            return FakeResp()

    class FakeAnthropic:
        def __init__(self, *args, **kwargs):
            self.messages = FakeMessages()

    monkeypatch.setenv("LLM_PROVIDER", "claude")
    monkeypatch.setattr("core.llm_client.Anthropic", FakeAnthropic)

    llm = get_sync_llm_client()
    out = llm.chat(messages=[{"role": "user", "content": "hi"}], model="claude-3", temperature=0.5)
    assert out == "hello"


def test_gemini_client(monkeypatch):
    class FakeGenResponse:
        def __init__(self):
            self.text = "gemini"
            self.usage_metadata = None

    class FakeModel:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
        def generate_content(self, messages, generation_config=None, request_options=None):
            self.messages = messages
            self.config = generation_config
            self.request_options = request_options
            return FakeGenResponse()

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    # Avoid real config
    monkeypatch.setattr("core.llm_client.genai.GenerativeModel", FakeModel)
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")

    llm = get_sync_llm_client()
    out = llm.chat(messages=[{"role": "user", "content": "hi"}], model="gemini-1.5-pro", temperature=0.2)
    assert out == "gemini"
