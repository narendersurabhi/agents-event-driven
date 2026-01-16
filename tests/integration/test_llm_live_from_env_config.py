"""Live integration test for the *configured* LLM in .env/config.

This validates that:
- `LLM_PROVIDER`, `LLM_MODEL`, and `LLM_TIMEOUT_SECONDS` are configured
- The provider-specific API key is present in config
- A minimal chat call succeeds end-to-end

Enable by setting `PYTEST_LLM_LIVE_CONFIG=1` (or `PYTEST_LLM_LIVE=1`) in your
environment or `.env`.
"""

from __future__ import annotations

import pytest

from core import config
from core.llm_factory import get_sync_llm_client

try:
    from google.api_core import exceptions as google_exceptions
except ImportError:  # pragma: no cover - optional dependency for non-gemini envs
    google_exceptions = None


def _is_live_enabled() -> bool:
    # Read via config so `.env` values work without `export`.
    return bool(
        config.get_config_value("PYTEST_LLM_LIVE_CONFIG")
        or config.get_config_value("PYTEST_LLM_LIVE")
    )


def _api_key_name(provider: str) -> str:
    provider = provider.lower()
    if provider.startswith("openai"):
        return "OPENAI_API_KEY"
    if provider == "claude":
        return "ANTHROPIC_API_KEY"
    if provider == "gemini":
        return "GOOGLE_API_KEY"
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}'")


def test_llm_live_from_env_config(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force using repo .env values even if the shell exported overrides.
    monkeypatch.setenv("DOTENV_PATH", ".env")
    config._config_adapter.cache_clear()

    if not _is_live_enabled():
        pytest.skip(
            "Live LLM test disabled; set PYTEST_LLM_LIVE_CONFIG=1 (or PYTEST_LLM_LIVE=1) to enable"
        )

    for key in [
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_TIMEOUT_SECONDS",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)
    config._config_adapter.cache_clear()

    provider = (config.get_config_value("LLM_PROVIDER", "openai") or "openai").lower()
    model = config.get_default_model()
    _ = config.get_timeout_seconds()  # validates config is present/parseable

    api_key_name = _api_key_name(provider)
    if not config.get_config_value(api_key_name):
        pytest.fail(f"{api_key_name} is not configured in .env for LLM_PROVIDER={provider}")

    llm = get_sync_llm_client(provider=provider)

    try:
        resp = llm.chat(
            messages=[
                {"role": "system", "content": "Reply with a single word."},
                {"role": "user", "content": "ping"},
            ],
            model=model,
            temperature=1.0,
        )
        print(f"LLM live test response: {resp}")
    except Exception as exc:
        # Make common Gemini misconfig clearer.
        if google_exceptions is not None and isinstance(exc, google_exceptions.NotFound):
            pytest.fail(f"Gemini model not found: LLM_MODEL={model}")
        raise

    assert isinstance(resp, str) and resp.strip()
