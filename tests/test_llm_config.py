import os

import pytest

from core import config


@pytest.fixture(autouse=True)
def clear_config_cache(monkeypatch):
    # Ensure each test sees fresh config based on its env.
    from core import config as cfg

    # Prevent tests from accidentally reading your real .env file.
    monkeypatch.setenv("DOTENV_PATH", "tests/.env.DO_NOT_USE")
    cfg._config_adapter.cache_clear()  # type: ignore[attr-defined]
    for key in ["LLM_MODEL", "LLM_TIMEOUT_SECONDS"]:
        monkeypatch.delenv(key, raising=False)
    yield
    cfg._config_adapter.cache_clear()  # type: ignore[attr-defined]


def test_get_default_model_requires_llm_model(monkeypatch):
    # No LLM_MODEL → must raise, no implicit defaults.
    with pytest.raises(RuntimeError):
        config.get_default_model()


def test_get_default_model_reads_llm_model(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gemini-1.5-pro")
    assert config.get_default_model() == "gemini-1.5-pro"


def test_get_timeout_seconds_requires_config(monkeypatch):
    with pytest.raises(RuntimeError):
        config.get_timeout_seconds()


def test_get_timeout_seconds_parses_value(monkeypatch):
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "90")
    assert config.get_timeout_seconds() == 90.0
