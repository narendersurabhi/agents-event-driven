""" Configuration management for the application."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

from core.config_adapter import (
    ConfigAdapter,
    ConfigSource,
    DotEnvConfigSource,
    EnvConfigSource,
    SecretsManagerConfigSource,
)


@lru_cache
def _config_adapter() -> ConfigAdapter:
    sources: list[ConfigSource] = [EnvConfigSource()]
    dotenv_path = Path(os.getenv("DOTENV_PATH", ".env"))
    sources.append(DotEnvConfigSource(path=dotenv_path))
    secret_id = os.getenv("AWS_SECRETSMANAGER_CONFIG_ID")
    if secret_id:
        sources.append(
            SecretsManagerConfigSource(
                secret_id=secret_id,
                region_name=os.getenv("AWS_REGION"),
                profile_name=os.getenv("AWS_PROFILE"),
            )
        )
    return ConfigAdapter(tuple(sources))


def get_config_value(key: str, default: str | None = None) -> str | None:
    return _config_adapter().get(key, default)


def get_default_model() -> str:
    """Return the configured default model name.

    Requires LLM_MODEL to be set in configuration; no implicit defaults.
    """
    value = get_config_value("LLM_MODEL")
    if not value:
        raise RuntimeError("LLM_MODEL is not configured; set it in your config/.env")
    return value


def get_timeout_seconds() -> float:
    """Return the configured timeout (seconds) for LLM calls.

    Requires LLM_TIMEOUT_SECONDS to be set in configuration.
    """
    raw = get_config_value("LLM_TIMEOUT_SECONDS")
    if not raw:
        raise RuntimeError("LLM_TIMEOUT_SECONDS is not configured; set it in your config/.env")
    try:
        return float(raw)
    except ValueError:
        return 60.0
