"""Centralized application settings for configuration-driven components."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache

from core.config import get_config_value


def _parse_csv(value: str | None, *, fallback: Iterable[str]) -> list[str]:
    if not value:
        return list(fallback)
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(slots=True)
class AppSettings:
    app_env: str = "dev"
    service_name: str = "tailor-api"
    app_version: str = "0.1.0"
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:3000"])

    def cors_allowlist(self) -> list[str]:
        if self.app_env == "dev" and not self.cors_origins:
            return ["*"]
        return self.cors_origins


@lru_cache
def get_app_settings() -> AppSettings:
    return AppSettings(
        app_env=get_config_value("APP_ENV", "dev") or "dev",
        service_name=get_config_value("SERVICE_NAME", "tailor-api") or "tailor-api",
        app_version=get_config_value("APP_VERSION", "0.1.0") or "0.1.0",
        cors_origins=_parse_csv(
            get_config_value("CORS_ORIGINS"), fallback=("http://localhost:3000",)
        ),
    )
