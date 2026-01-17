"""Structured logging utilities."""

from __future__ import annotations

from datetime import datetime
import json
import logging
from typing import Any


class JsonFormatter(logging.Formatter):
    """Render log records as structured JSON."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)
        return json.dumps(payload)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure JSON logging for the service."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=level, handlers=[handler])
