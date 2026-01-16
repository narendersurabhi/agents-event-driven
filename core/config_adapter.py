""" Configuration adapter implementations for various sources. """

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Protocol

try:
    import boto3
except ImportError:  # pragma: no cover - optional dependency for local usage
    boto3 = None


class ConfigSource(Protocol):
    """Strategy interface for pulling configuration values from a backing store."""

    def get(self, key: str) -> str | None: ...


@dataclass(slots=True)
class EnvConfigSource:
    """Reads values directly from environment variables."""

    prefix: str | None = None

    def get(self, key: str) -> str | None:
        env_key = f"{self.prefix}{key}" if self.prefix else key
        return os.getenv(env_key)


@dataclass(slots=True)
class DotEnvConfigSource:
    """Lightweight .env reader so we avoid extra dependencies."""

    path: Path = Path(".env")
    encoding: str = "utf-8"
    _cache: dict[str, str] = field(default_factory=dict, init=False)
    _loaded: bool = field(default=False, init=False)

    def _load(self) -> None:
        if self._loaded:
            return
        try:
            for raw_line in self.path.read_text(encoding=self.encoding).splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                self._cache[key.strip()] = self._strip_quotes(value.strip())
        except FileNotFoundError:
            pass
        finally:
            self._loaded = True

    @staticmethod
    def _strip_quotes(value: str) -> str:
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            return value[1:-1]
        return value

    def get(self, key: str) -> str | None:
        self._load()
        return self._cache.get(key)


@dataclass(slots=True)
class SecretsManagerConfigSource:
    """Reads configuration entries from AWS Secrets Manager JSON payloads."""

    secret_id: str
    region_name: str | None = None
    profile_name: str | None = None
    _cache: dict[str, str] = field(default_factory=dict, init=False)
    _loaded: bool = field(default=False, init=False)

    def _client(self) -> Any:
        if boto3 is None:
            raise RuntimeError("boto3 is required for SecretsManagerConfigSource")
        session = (
            boto3.session.Session(profile_name=self.profile_name)
            if self.profile_name
            else boto3.session.Session()
        )
        return session.client("secretsmanager", region_name=self.region_name)

    def _load(self) -> None:
        if self._loaded:
            return
        client = self._client()
        resp = client.get_secret_value(SecretId=self.secret_id)
        secret_string = resp.get("SecretString")
        if not secret_string:
            binary_secret = resp.get("SecretBinary")
            if binary_secret:
                secret_string = base64.b64decode(binary_secret).decode("utf-8")
        if not secret_string:
            self._loaded = True
            return
        try:
            parsed = json.loads(secret_string)
            if isinstance(parsed, dict):
                self._cache.update({k: str(v) for k, v in parsed.items()})
            else:
                self._cache["SECRET_STRING"] = str(parsed)
        except json.JSONDecodeError:
            self._cache["SECRET_STRING"] = secret_string
        finally:
            self._loaded = True

    def get(self, key: str) -> str | None:
        self._load()
        return self._cache.get(key)


@dataclass(slots=True)
class ConfigAdapter:
    """Composite over multiple sources (env → .env → secrets)."""

    sources: tuple[ConfigSource, ...]

    def get(self, key: str, default: str | None = None) -> str | None:
        for source in self.sources:
            value = source.get(key)
            if value is not None:
                return value
        return default
