"""Candidate resume context loader for prompt alignment."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from core.config import get_config_value


@lru_cache
def get_candidate_context() -> str | None:
    """Return optional candidate context text for aligning agent prompts."""
    text = get_config_value("CANDIDATE_PROFILE_TEXT")
    if text is not None:
        stripped = text.strip()
        if stripped:
            return stripped

    path_value = get_config_value("CANDIDATE_PROFILE_PATH")
    if not path_value:
        return None

    path = Path(path_value).expanduser()
    if not path.is_file():
        return None

    content = path.read_text(encoding="utf-8").strip()
    return content or None
