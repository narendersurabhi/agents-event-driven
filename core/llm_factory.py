"""Factory functions to get LLM clients based on configuration."""

from __future__ import annotations

from typing import Callable, Dict

from core.config import get_config_value, get_timeout_seconds
from core.llm_client import (
    AsyncLLMClient,
    LLMClient,
    AsyncOpenAILLMClient,
    OpenAILLMClient,
    AsyncOpenAIGPT5LLMClient,
    OpenAIGPT5LLMClient,
    AsyncClaudeLLMClient,
    ClaudeLLMClient,
    AsyncGeminiLLMClient,
    GeminiLLMClient,
)
from core.obs import Logger, JsonRepoLogger


# Provider registry for simple DI. Extend as new adapters are added.
_ASYNC_PROVIDERS: Dict[str, Callable[[Logger, float], AsyncLLMClient]] = {
    "openai": lambda logger, timeout: AsyncOpenAILLMClient(logger=logger, timeout=timeout),
    "openai-gpt5": lambda logger, timeout: AsyncOpenAIGPT5LLMClient(logger=logger, timeout=timeout),
    "claude": lambda logger, timeout: AsyncClaudeLLMClient(logger=logger, timeout=timeout),
    "gemini": lambda logger, timeout: AsyncGeminiLLMClient(logger=logger, timeout=timeout),
}

_SYNC_PROVIDERS: Dict[str, Callable[[Logger, float], LLMClient]] = {
    "openai": lambda logger, timeout: OpenAILLMClient(logger=logger, timeout=timeout),
    "openai-gpt5": lambda logger, timeout: OpenAIGPT5LLMClient(logger=logger, timeout=timeout),
    "claude": lambda logger, timeout: ClaudeLLMClient(logger=logger, timeout=timeout),
    "gemini": lambda logger, timeout: GeminiLLMClient(logger=logger, timeout=timeout),
}


def get_async_llm_client(logger: Logger | None = None, provider: str | None = None) -> AsyncLLMClient:
    # Use a shared JSON repo logger by default so all LLM calls are observable.
    logger = logger or JsonRepoLogger(service="llm")
    name = (provider or get_config_value("LLM_PROVIDER", "openai") or "openai").lower()
    timeout = get_timeout_seconds()
    try:
        return _ASYNC_PROVIDERS[name](logger, timeout)
    except KeyError as exc:
        raise ValueError(f"Unknown async LLM provider '{name}'") from exc


def get_sync_llm_client(logger: Logger | None = None, provider: str | None = None) -> LLMClient:
    # Use a shared JSON repo logger by default so all LLM calls are observable.
    logger = logger or JsonRepoLogger(service="llm")
    name = (provider or get_config_value("LLM_PROVIDER", "openai") or "openai").lower()
    timeout = get_timeout_seconds()
    try:
        return _SYNC_PROVIDERS[name](logger, timeout)
    except KeyError as exc:
        raise ValueError(f"Unknown sync LLM provider '{name}'") from exc
