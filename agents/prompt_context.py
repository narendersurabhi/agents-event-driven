"""Utilities for adding candidate context to prompts."""

from __future__ import annotations

from core.candidate_context import get_candidate_context


def append_candidate_context(content: str) -> str:
    """Append candidate context to the provided prompt content if available."""
    context = get_candidate_context()
    if not context:
        return content
    return f"{content}\n\nCANDIDATE CONTEXT (align to this resume):\n{context}\n"
