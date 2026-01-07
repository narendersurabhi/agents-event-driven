""" Integration tests for live LLM clients."""

import os
import pytest
from google.api_core import exceptions as google_exceptions

from core.llm_factory import get_async_llm_client, get_sync_llm_client


LIVE_FLAG = os.getenv("PYTEST_LLM_LIVE")

if not LIVE_FLAG:
    pytest.skip("live LLM test disabled; set PYTEST_LLM_LIVE=1 to enable", allow_module_level=True)


def _has_key(provider: str) -> bool:
    if provider.startswith("openai"):
        return bool(os.getenv("OPENAI_API_KEY"))
    if provider == "claude":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    if provider == "gemini":
        return bool(os.getenv("GOOGLE_API_KEY"))
    return False


def _default_model(provider: str) -> str:
    if provider.startswith("openai"):
        return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    if provider == "claude":
        return os.getenv("CLAUDE_MODEL", "claude-3-sonnet-20240229")
    if provider == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-1.0-pro")
    return "gpt-4.1-mini"


@pytest.mark.parametrize("provider", ["openai", "openai-gpt5", "claude", "gemini"])
@pytest.mark.asyncio
async def test_async_llm_live(provider):
    if not _has_key(provider):
        pytest.skip(f"No API key for provider {provider}")
    llm = get_async_llm_client(provider=provider)
    try:
        resp = await llm.chat(
            messages=[{"role": "system", "content": "You are terse."}, {"role": "user", "content": "ping"}],
            model=_default_model(provider),
            temperature=1.0,
        )
    except google_exceptions.NotFound:
        pytest.skip(f"Gemini model not found; set GEMINI_MODEL to a valid model for provider {provider}")
    assert isinstance(resp, str) and resp.strip()


@pytest.mark.parametrize("provider", ["openai", "openai-gpt5", "claude", "gemini"])
def test_sync_llm_live(provider):
    if not _has_key(provider):
        pytest.skip(f"No API key for provider {provider}")
    llm = get_sync_llm_client(provider=provider)
    try:
        resp = llm.chat(
            messages=[{"role": "system", "content": "You are terse."}, {"role": "user", "content": "ping"}],
            model=_default_model(provider),
            temperature=1.0,
        )
    except google_exceptions.NotFound:
        pytest.skip(f"Gemini model not found; set GEMINI_MODEL to a valid model for provider {provider}")
    assert isinstance(resp, str) and resp.strip()
