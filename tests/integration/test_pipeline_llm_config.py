import pytest

from api import pipeline
from core import config


def _reset_runtime():
    # Reset cached runtime between tests so _ensure_runtime re-evaluates config.
    pipeline._runtime = None  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def isolate_dotenv(monkeypatch):
    # Prevent tests from accidentally reading your real .env file.
    monkeypatch.setenv("DOTENV_PATH", "tests/.env.DO_NOT_USE")
    config._config_adapter.cache_clear()  # type: ignore[attr-defined]
    yield
    config._config_adapter.cache_clear()  # type: ignore[attr-defined]


def test_pipeline_ensure_runtime_uses_config(monkeypatch):
    # With valid config, _ensure_runtime should succeed and return a runtime.
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-1.5-pro")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    config._config_adapter.cache_clear()  # type: ignore[attr-defined]
    _reset_runtime()

    runtime = pipeline._ensure_runtime()
    assert runtime is not None
    assert runtime.bus is not None
    assert runtime.orchestrator is not None


def test_pipeline_ensure_runtime_fails_without_model(monkeypatch):
    # Missing LLM_MODEL should cause get_default_model to raise.
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "120")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    config._config_adapter.cache_clear()  # type: ignore[attr-defined]
    _reset_runtime()

    with pytest.raises(RuntimeError):
        pipeline._ensure_runtime()
