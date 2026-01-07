""" LLM client ports and adapters. """

from __future__ import annotations

import logging
import os, time, random, asyncio
import httpx

from typing import Any, Callable, Optional, Protocol
import openai
from openai import OpenAI, AsyncOpenAI
import anthropic
from anthropic import Anthropic, AsyncAnthropic
import google.generativeai as genai
from core.config import get_config_value
from core.obs import Logger, NullLogger, with_span
import uuid

logger = logging.getLogger(__name__)


def _parse_int(raw: str) -> int | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        try:
            return int(float(raw))
        except ValueError:
            return None


def _get_int_setting(*keys: str, default: int | None = None) -> int | None:
    for key in keys:
        raw = get_config_value(key)
        if raw is None:
            continue
        parsed = _parse_int(raw)
        if parsed is not None:
            return parsed
    return default


def _get_float_setting(key: str, default: float) -> float:
    raw = get_config_value(key)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _normalize_temperature(model: str, temperature: float | None, logger: Logger, req_id: str) -> float | None:
    """For models that disallow custom temps (e.g., gpt-5*), drop it unless exactly 1."""
    if model.lower().startswith("gpt-5"):
        if temperature is not None and temperature != 1:
            logger.warn(
                "llm.temperature_ignored",
                req_id=req_id,
                model=model,
                requested=temperature,
                reason="gpt-5 only supports default temperature",
            )
        return None
    return temperature if temperature is not None else 0.0


def _ensure_req_id(_args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    req_id = kwargs.get("req_id")
    if not isinstance(req_id, str) or not req_id:
        kwargs["req_id"] = str(uuid.uuid4())


def _llm_span_fields(*args: Any, **kwargs: Any) -> dict[str, Any]:
    model = kwargs.get("model")
    if model is None and len(args) >= 3:
        model = args[2]
    return {"req_id": kwargs.get("req_id"), "model": model}


def _llm_span(provider: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return with_span(
        "llm.chat",
        logger_attr="_logger",
        fields={"provider": provider},
        fields_fn=_llm_span_fields,
        pre=_ensure_req_id,
    )


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _log_content_enabled() -> bool:
    """Whether logs may include prompt/response text previews.

    Defaults to enabled to preserve current dev behavior; disable by setting
    `LLM_LOG_CONTENT=0` in config/.env.
    """

    raw = get_config_value("LLM_LOG_CONTENT")
    if raw is None:
        return True
    return _is_truthy(raw)


def _safe_text_preview(text: str, limit: int = 1000) -> str:
    return (text or "")[:limit]


def _safe_openai_messages(messages: list[dict[str, str]], *, log_content: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        entry: dict[str, Any] = {"role": role}
        if log_content:
            entry["content"] = _safe_text_preview(content)
        else:
            entry["content_len"] = len(content)
        out.append(entry)
    return out


def _safe_anthropic_messages(messages: list[dict[str, str]], *, log_content: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        entry: dict[str, Any] = {"role": role}
        if log_content:
            entry["content"] = _safe_text_preview(content)
        else:
            entry["content_len"] = len(content)
        out.append(entry)
    return out


def _safe_gemini_messages(messages: list[dict[str, Any]], *, log_content: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        parts = list(m.get("parts") or [])
        entry: dict[str, Any] = {"role": role}
        if log_content:
            entry["parts"] = [_safe_text_preview(str(p or "")) for p in parts]
        else:
            entry["parts_len"] = [len(str(p or "")) for p in parts]
        out.append(entry)
    return out


# ---------- Sync port ----------


class LLMClient(Protocol):
    """Port interface for synchronous LLM calls."""

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
    ) -> str:
        """
        Send chat messages to an LLM and return the assistant's content.
        """
        ...


class OpenAILLMClient(LLMClient):
    def __init__(
        self,
        timeout: float | None = None,
        max_retries: int = 3,
        logger: Optional[Logger] = None,
        api_key: str | None = None,
    ):
        self._timeout = float(timeout) if timeout is not None else _get_float_setting("LLM_TIMEOUT_SECONDS", 120.0)
        self._max_retries = max_retries
        api_key = api_key or get_config_value("OPENAI_API_KEY")
        self._client = OpenAI(api_key=api_key)
        self._logger: Logger = logger or NullLogger()
        tokens = _get_int_setting("OPENAI_MAX_COMPLETION_TOKENS", "OPENAI_MAX_OUTPUT_TOKENS", default=0)
        self._default_max_completion_tokens = tokens or None

    @_llm_span("openai")
    def chat(self, messages, model, temperature=0.0, **kwargs) -> str:
        req_id = kwargs.pop("req_id", str(uuid.uuid4()))
        log_content = _log_content_enabled()
        safe_messages = _safe_openai_messages(messages, log_content=log_content)
        temp_for_request = _normalize_temperature(model, temperature, self._logger, req_id)
        self._logger.info(
            "llm.request",
            req_id=req_id,
            provider="openai",
            model=model,
            temperature=temperature,
            kwargs=kwargs,
            message_count=len(messages),
            messages=safe_messages,
        )
        for attempt in range(self._max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    timeout=self._timeout,
                    temperature=temp_for_request,
                    **kwargs,
                )
                content = resp.choices[0].message.content
                usage = getattr(resp, "usage", None)
                resp_fields: dict[str, Any] = {
                    "req_id": req_id,
                    "provider": "openai",
                    "model": model,
                    "attempt": attempt + 1,
                    "usage": getattr(usage, "__dict__", None) if usage else None,
                    "content_len": len(content or ""),
                }
                if log_content:
                    resp_fields["preview"] = _safe_text_preview(content or "")
                self._logger.info("llm.response", **resp_fields)
                return content
            except (openai.APITimeoutError, httpx.ReadTimeout) as e:
                self._logger.warn(
                    "llm.timeout",
                    req_id=req_id,
                    provider="openai",
                    model=model,
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt == self._max_retries - 1:
                    raise
                time.sleep(2 ** attempt + random.random())


# ---------- Async port ----------


class AsyncLLMClient(Protocol):
    """Port interface for async LLM calls."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
    ) -> str:
        """
        Send chat messages to an LLM and return the assistant's content.
        """
        ...


class AsyncOpenAILLMClient(AsyncLLMClient):
    def __init__(
        self,
        timeout: float | None = None,
        max_retries: int = 3,
        logger: Optional[Logger] = None,
        api_key: str | None = None,
    ):
        self._timeout = float(timeout) if timeout is not None else _get_float_setting("LLM_TIMEOUT_SECONDS", 120.0)
        self._max_retries = max_retries
        api_key = api_key or get_config_value("OPENAI_API_KEY")
        self._client = AsyncOpenAI(api_key=api_key)
        self._logger: Logger = logger or NullLogger()

    @_llm_span("openai")
    async def chat(self, messages, model, temperature=0.0, **kwargs) -> str:
        req_id = kwargs.pop("req_id", str(uuid.uuid4()))
        log_content = _log_content_enabled()
        safe_messages = _safe_openai_messages(messages, log_content=log_content)
        temp_for_request = _normalize_temperature(model, temperature, self._logger, req_id)

        async def _attempt(attempt_num: int) -> str:
            resp = await self._client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=self._timeout,
                temperature=temp_for_request,
                **kwargs,
            )
            content = resp.choices[0].message.content
            usage = getattr(resp, "usage", None)
            resp_fields: dict[str, Any] = {
                "req_id": req_id,
                "provider": "openai",
                "model": model,
                "attempt": attempt_num,
                "usage": getattr(usage, "__dict__", None) if usage else None,
                "content_len": len(content or ""),
            }
            if log_content:
                resp_fields["preview"] = _safe_text_preview(content or "")
            self._logger.info("llm.response", **resp_fields)
            return content

        self._logger.info(
            "llm.request",
            req_id=req_id,
            provider="openai",
            model=model,
            temperature=temperature,
            kwargs=kwargs,
            message_count=len(messages),
            messages=safe_messages,
        )
        for attempt in range(self._max_retries):
            try:
                return await _attempt(attempt + 1)
            except (openai.APITimeoutError, httpx.ReadTimeout) as e:
                self._logger.warn(
                    "llm.timeout",
                    req_id=req_id,
                    provider="openai",
                    model=model,
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt == self._max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt + random.random())


# ---------- GPT-5 specific clients (omit unsupported params like temperature) ----------


class OpenAIGPT5LLMClient(LLMClient):
    """OpenAI client variant that avoids passing unsupported params to GPT-5 models."""

    def __init__(
        self,
        timeout: float | None = None,
        max_retries: int = 3,
        logger: Optional[Logger] = None,
        api_key: str | None = None,
    ):
        self._timeout = float(timeout) if timeout is not None else _get_float_setting("LLM_TIMEOUT_SECONDS", 120.0)
        self._max_retries = max_retries
        api_key = api_key or get_config_value("OPENAI_API_KEY")
        self._client = OpenAI(api_key=api_key)
        self._logger: Logger = logger or NullLogger()
        tokens = _get_int_setting("OPENAI_MAX_COMPLETION_TOKENS", "OPENAI_MAX_OUTPUT_TOKENS", default=0)
        self._default_max_completion_tokens = tokens or None

    @_llm_span("openai")
    def chat(self, messages, model, temperature: float | None = None, **kwargs) -> str:
        req_id = kwargs.pop("req_id", str(uuid.uuid4()))
        log_content = _log_content_enabled()
        safe_messages = _safe_openai_messages(messages, log_content=log_content)

        temp_for_request = _normalize_temperature(model, temperature, self._logger, req_id)

        def _payload():
            payload = dict(model=model, messages=messages, timeout=self._timeout, **kwargs)
            max_tokens = (
                payload.pop("max_completion_tokens", None)
                or payload.pop("max_output_tokens", None)
                or payload.pop("max_tokens", None)
                or self._default_max_completion_tokens
            )
            if max_tokens:
                payload["max_completion_tokens"] = max_tokens
            if temp_for_request is not None:
                payload["temperature"] = temp_for_request
            return payload

        self._logger.info(
            "llm.request",
            req_id=req_id,
            provider="openai",
            model=model,
            temperature=temperature,
            kwargs=kwargs,
            message_count=len(messages),
            messages=safe_messages,
        )
        last_err = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.chat.completions.create(**_payload())
                content = resp.choices[0].message.content
                usage = getattr(resp, "usage", None)
                resp_fields: dict[str, Any] = {
                    "req_id": req_id,
                    "provider": "openai",
                    "model": model,
                    "attempt": attempt + 1,
                    "usage": getattr(usage, "__dict__", None) if usage else None,
                    "content_len": len(content or ""),
                }
                if log_content:
                    resp_fields["preview"] = _safe_text_preview(content or "")
                self._logger.info("llm.response", **resp_fields)
                return content
            except (openai.APITimeoutError, httpx.ReadTimeout) as e:
                last_err = e
                self._logger.warn(
                    "llm.timeout",
                    req_id=req_id,
                    provider="openai",
                    model=model,
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt == self._max_retries - 1: raise
                time.sleep(2 ** attempt + random.random())


class AsyncOpenAIGPT5LLMClient(AsyncLLMClient):
    """Async variant that avoids unsupported params for GPT-5 models."""

    def __init__(
        self,
        timeout: float | None = None,
        max_retries: int = 3,
        logger: Optional[Logger] = None,
        api_key: str | None = None,
    ):
        self._timeout = float(timeout) if timeout is not None else _get_float_setting("LLM_TIMEOUT_SECONDS", 120.0)
        self._max_retries = max_retries
        api_key = api_key or get_config_value("OPENAI_API_KEY")
        self._client = AsyncOpenAI(api_key=api_key)
        self._logger: Logger = logger or NullLogger()
        tokens = _get_int_setting("OPENAI_MAX_COMPLETION_TOKENS", "OPENAI_MAX_OUTPUT_TOKENS", default=0)
        self._default_max_completion_tokens = tokens or None

    @_llm_span("openai")
    async def chat(self, messages, model, temperature: float | None = None, **kwargs) -> str:
        req_id = kwargs.pop("req_id", str(uuid.uuid4()))
        log_content = _log_content_enabled()
        safe_messages = _safe_openai_messages(messages, log_content=log_content)

        temp_for_request = _normalize_temperature(model, temperature, self._logger, req_id)

        def _payload():
            payload = dict(model=model, messages=messages, timeout=self._timeout, **kwargs)
            max_tokens = (
                payload.pop("max_completion_tokens", None)
                or payload.pop("max_output_tokens", None)
                or payload.pop("max_tokens", None)
                or self._default_max_completion_tokens
            )
            if max_tokens:
                payload["max_completion_tokens"] = max_tokens
            if temp_for_request is not None:
                payload["temperature"] = temp_for_request
            return payload

        async def _attempt(attempt_num: int) -> str:
            resp = await self._client.chat.completions.create(**_payload())
            content = resp.choices[0].message.content
            usage = getattr(resp, "usage", None)
            resp_fields: dict[str, Any] = {
                "req_id": req_id,
                "provider": "openai",
                "model": model,
                "attempt": attempt_num,
                "usage": getattr(usage, "__dict__", None) if usage else None,
                "content_len": len(content or ""),
            }
            if log_content:
                resp_fields["preview"] = _safe_text_preview(content or "")
            self._logger.info("llm.response", **resp_fields)
            return content

        self._logger.info(
            "llm.request",
            req_id=req_id,
            provider="openai",
            model=model,
            temperature=temperature,
            kwargs=kwargs,
            message_count=len(messages),
            messages=safe_messages,
        )
        last_err = None
        for attempt in range(self._max_retries):
            try:
                return await _attempt(attempt + 1)
            except (openai.APITimeoutError, httpx.ReadTimeout) as e:
                last_err = e
                self._logger.warn(
                    "llm.timeout",
                    req_id=req_id,
                    provider="openai",
                    model=model,
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt == self._max_retries - 1: raise
                await asyncio.sleep(2 ** attempt + random.random())


# ---------- Anthropic Claude clients ----------


def _split_anthropic_messages(messages: list[dict[str, str]]):
    system = None
    converted: list[dict[str, str]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system" and system is None:
            system = content
            continue
        if role == "assistant":
            converted.append({"role": "assistant", "content": content})
        else:
            converted.append({"role": "user", "content": content})
    return system, converted


class ClaudeLLMClient(LLMClient):
    """Sync Claude client implementing the LLMClient protocol."""

    def __init__(
        self,
        timeout: float | None = None,
        logger: Optional[Logger] = None,
        max_tokens: int | None = None,
        api_key: str | None = None,
    ):
        timeout_value = float(timeout) if timeout is not None else _get_float_setting("LLM_TIMEOUT_SECONDS", 120.0)
        api_key = api_key or get_config_value("ANTHROPIC_API_KEY")
        self._client = Anthropic(api_key=api_key, timeout=timeout_value)
        self._logger: Logger = logger or NullLogger()
        self._max_tokens = max_tokens or _get_int_setting("CLAUDE_MAX_TOKENS", default=1024) or 1024

    @_llm_span("anthropic")
    def chat(self, messages, model, temperature=1.0, **kwargs) -> str:
        req_id = kwargs.pop("req_id", str(uuid.uuid4()))
        system, converted = _split_anthropic_messages(messages)
        log_content = _log_content_enabled()
        safe_converted = _safe_anthropic_messages(converted, log_content=log_content)
        safe_system = _safe_text_preview(system) if (log_content and system) else None
        system_len = len(system) if system else 0
        payload = {
            "model": model,
            "messages": converted,
            "max_tokens": kwargs.pop("max_tokens", self._max_tokens),
            "temperature": temperature,
        }
        if system:
            payload["system"] = system
        self._logger.info(
            "llm.request",
            req_id=req_id,
            provider="anthropic",
            model=model,
            kwargs=kwargs,
            message_count=len(converted),
            messages=safe_converted,
            system=safe_system,
            system_len=system_len,
        )
        resp = self._client.messages.create(**payload)
        content = resp.content[0].text if resp.content else ""
        usage = getattr(resp, "usage", None)
        resp_fields: dict[str, Any] = {
            "req_id": req_id,
            "provider": "anthropic",
            "model": model,
            "attempt": 1,
            "usage": getattr(usage, "__dict__", None) if usage else None,
            "content_len": len(content or ""),
        }
        if log_content:
            resp_fields["preview"] = _safe_text_preview(content or "")
        self._logger.info("llm.response", **resp_fields)
        return content


# ---------- Google Gemini clients ----------


def _split_gemini_messages(messages: list[dict[str, str]]):
    system = None
    converted = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system" and system is None:
            system = content
            continue
        if role == "assistant":
            converted.append({"role": "model", "parts": [content]})
        else:
            converted.append({"role": "user", "parts": [content]})
    return system, converted


def _finish_reason_to_str(reason: object) -> str:
    name = getattr(reason, "name", None)
    if isinstance(name, str) and name:
        return name
    try:
        return str(reason)
    except Exception:  # pragma: no cover
        return "<unknown>"


def _extract_gemini_text(resp: object) -> str:
    """Best-effort extraction of text from Gemini responses.

    `google.generativeai` exposes a `response.text` accessor, but it raises if the
    response contains no text `Part` (e.g., blocked output, tool-only parts, etc).
    """
    try:
        # `GenerateContentResponse.text` may raise; keep it in a narrow try block.
        text = getattr(resp, "text")
        return (text or "").strip()
    except Exception:
        candidates = getattr(resp, "candidates", None) or []
        if candidates:
            cand0 = candidates[0]
            content = getattr(cand0, "content", None)
            parts = getattr(content, "parts", None) or []
            texts: list[str] = []
            for part in parts:
                t = getattr(part, "text", None)
                if t:
                    texts.append(str(t))
            if texts:
                return "\n".join(texts).strip()
            finish_reason = getattr(cand0, "finish_reason", None)
            raise RuntimeError(
                "Gemini returned no text parts (candidate finish_reason="
                f"{_finish_reason_to_str(finish_reason)}). "
                "If this is MAX_TOKENS, increase LLM_MAX_OUTPUT_TOKENS/GEMINI_MAX_TOKENS."
            )
        raise RuntimeError("Gemini returned no candidates / no text parts")


class GeminiLLMClient(LLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        logger: Optional[Logger] = None,
        max_output_tokens: int | None = None,
        timeout: float | None = None,
    ):
        self._logger: Logger = logger or NullLogger()
        api_key = api_key or get_config_value("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        self._max_tokens = max_output_tokens or _get_int_setting("GEMINI_MAX_TOKENS", "LLM_MAX_OUTPUT_TOKENS", default=8192) or 8192
        self._timeout = float(timeout) if timeout is not None else _get_float_setting("LLM_TIMEOUT_SECONDS", 120.0)

    @_llm_span("gemini")
    def chat(self, messages, model, temperature=1.0, **kwargs) -> str:
        req_id = kwargs.pop("req_id", str(uuid.uuid4()))
        system, converted = _split_gemini_messages(messages)
        log_content = _log_content_enabled()
        safe_converted = _safe_gemini_messages(converted, log_content=log_content)
        safe_system = _safe_text_preview(system) if (log_content and system) else None
        system_len = len(system) if system else 0
        gen_config = {"temperature": temperature, "max_output_tokens": kwargs.pop("max_output_tokens", self._max_tokens)}
        gm = genai.GenerativeModel(model_name=model, system_instruction=system)
        request_options = genai.types.RequestOptions(timeout=self._timeout) if self._timeout else None
        self._logger.info(
            "llm.request",
            req_id=req_id,
            provider="gemini",
            model=model,
            kwargs=kwargs,
            message_count=len(converted),
            messages=safe_converted,
            system=safe_system,
            system_len=system_len,
        )
        resp = gm.generate_content(converted, generation_config=gen_config, request_options=request_options)
        content = _extract_gemini_text(resp)
        usage = getattr(resp, "usage_metadata", None)
        resp_fields: dict[str, Any] = {
            "req_id": req_id,
            "provider": "gemini",
            "model": model,
            "attempt": 1,
            "usage": getattr(usage, "__dict__", None) if usage else None,
            "content_len": len(content or ""),
        }
        if log_content:
            resp_fields["preview"] = _safe_text_preview(content or "")
        self._logger.info("llm.response", **resp_fields)
        return content


class AsyncGeminiLLMClient(AsyncLLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        logger: Optional[Logger] = None,
        max_output_tokens: int | None = None,
        timeout: float | None = None,
    ):
        self._logger: Logger = logger or NullLogger()
        api_key = api_key or get_config_value("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        self._max_tokens = max_output_tokens or _get_int_setting("GEMINI_MAX_TOKENS", "LLM_MAX_OUTPUT_TOKENS", default=8192) or 8192
        self._timeout = float(timeout) if timeout is not None else _get_float_setting("LLM_TIMEOUT_SECONDS", 120.0)

    @_llm_span("gemini")
    async def chat(self, messages, model, temperature=1.0, **kwargs) -> str:
        req_id = kwargs.pop("req_id", str(uuid.uuid4()))
        system, converted = _split_gemini_messages(messages)
        log_content = _log_content_enabled()
        safe_converted = _safe_gemini_messages(converted, log_content=log_content)
        safe_system = _safe_text_preview(system) if (log_content and system) else None
        system_len = len(system) if system else 0
        gen_config = {"temperature": temperature, "max_output_tokens": kwargs.pop("max_output_tokens", self._max_tokens)}
        gm = genai.GenerativeModel(model_name=model, system_instruction=system)
        request_options = genai.types.RequestOptions(timeout=self._timeout) if self._timeout else None
        self._logger.info(
            "llm.request",
            req_id=req_id,
            provider="gemini",
            model=model,
            kwargs=kwargs,
            message_count=len(converted),
            messages=safe_converted,
            system=safe_system,
            system_len=system_len,
        )
        resp = await gm.generate_content_async(converted, generation_config=gen_config, request_options=request_options)
        content = _extract_gemini_text(resp)
        usage = getattr(resp, "usage_metadata", None)
        resp_fields: dict[str, Any] = {
            "req_id": req_id,
            "provider": "gemini",
            "model": model,
            "attempt": 1,
            "usage": getattr(usage, "__dict__", None) if usage else None,
            "content_len": len(content or ""),
        }
        if log_content:
            resp_fields["preview"] = _safe_text_preview(content or "")
        self._logger.info("llm.response", **resp_fields)
        return content


class AsyncClaudeLLMClient(AsyncLLMClient):
    """Async Claude client implementing the AsyncLLMClient protocol."""

    def __init__(
        self,
        timeout: float | None = None,
        logger: Optional[Logger] = None,
        max_tokens: int | None = None,
        api_key: str | None = None,
    ):
        timeout_value = float(timeout) if timeout is not None else _get_float_setting("LLM_TIMEOUT_SECONDS", 120.0)
        api_key = api_key or get_config_value("ANTHROPIC_API_KEY")
        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout_value)
        self._logger: Logger = logger or NullLogger()
        self._max_tokens = max_tokens or _get_int_setting("CLAUDE_MAX_TOKENS", default=1024) or 1024

    @_llm_span("anthropic")
    async def chat(self, messages, model, temperature=1.0, **kwargs) -> str:
        req_id = kwargs.pop("req_id", str(uuid.uuid4()))
        system, converted = _split_anthropic_messages(messages)
        log_content = _log_content_enabled()
        safe_converted = _safe_anthropic_messages(converted, log_content=log_content)
        safe_system = _safe_text_preview(system) if (log_content and system) else None
        system_len = len(system) if system else 0
        payload = {
            "model": model,
            "messages": converted,
            "max_tokens": kwargs.pop("max_tokens", self._max_tokens),
            "temperature": temperature,
        }
        if system:
            payload["system"] = system

        self._logger.info(
            "llm.request",
            req_id=req_id,
            provider="anthropic",
            model=model,
            kwargs=kwargs,
            message_count=len(converted),
            messages=safe_converted,
            system=safe_system,
            system_len=system_len,
        )
        resp = await self._client.messages.create(**payload)
        content = resp.content[0].text if resp.content else ""
        usage = getattr(resp, "usage", None)
        resp_fields: dict[str, Any] = {
            "req_id": req_id,
            "provider": "anthropic",
            "model": model,
            "attempt": 1,
            "usage": getattr(usage, "__dict__", None) if usage else None,
            "content_len": len(content or ""),
        }
        if log_content:
            resp_fields["preview"] = _safe_text_preview(content or "")
        self._logger.info("llm.response", **resp_fields)
        return content
