from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


class ApiError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class ApiClient:
    base_url: str
    timeout_short: float = 10.0
    timeout_long: float = 300.0
    timeout_docx: float = 60.0

    def _url(self, path: str) -> str:
        base = self.base_url.rstrip("/")
        cleaned = path if path.startswith("/") else f"/{path}"
        return f"{base}{cleaned}"

    def get_json(self, path: str, *, timeout: float | None = None) -> Any:
        url = self._url(path)
        try:
            resp = requests.get(url, timeout=timeout or self.timeout_short)
        except Exception as exc:  # noqa: BLE001
            raise ApiError(f"Request failed: {exc}") from exc
        if resp.status_code != 200:
            raise ApiError(
                f"API returned {resp.status_code}: {resp.text}", status_code=resp.status_code
            )
        return resp.json()

    def post_json(
        self, path: str, *, payload: dict[str, Any] | None = None, timeout: float | None = None
    ) -> Any:
        url = self._url(path)
        try:
            resp = requests.post(url, json=payload, timeout=timeout or self.timeout_long)
        except Exception as exc:  # noqa: BLE001
            raise ApiError(f"Request failed: {exc}") from exc
        if resp.status_code != 200:
            raise ApiError(
                f"API returned {resp.status_code}: {resp.text}", status_code=resp.status_code
            )
        return resp.json()

    def get_bytes(self, path: str, *, timeout: float | None = None) -> bytes:
        url = self._url(path)
        try:
            resp = requests.get(url, timeout=timeout or self.timeout_docx)
        except Exception as exc:  # noqa: BLE001
            raise ApiError(f"Request failed: {exc}") from exc
        if resp.status_code != 200:
            raise ApiError(
                f"API returned {resp.status_code}: {resp.text}", status_code=resp.status_code
            )
        return resp.content
