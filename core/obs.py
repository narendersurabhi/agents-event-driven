from __future__ import annotations

import datetime
import functools
import inspect
import json
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol

from core.config import get_config_value

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = REPO_ROOT / "logs"

_FILE_LOCKS: dict[Path, threading.Lock] = {}
_FILE_LOCKS_GUARD = threading.Lock()


def _file_lock(path: Path) -> threading.Lock:
    with _FILE_LOCKS_GUARD:
        lock = _FILE_LOCKS.get(path)
        if lock is None:
            lock = threading.Lock()
            _FILE_LOCKS[path] = lock
        return lock


class Logger(Protocol):
    def info(self, event: str, **fields: Any) -> None: ...
    def warn(self, event: str, **fields: Any) -> None: ...
    def error(self, event: str, **fields: Any) -> None: ...

class NullLogger:
    def info(self, event: str, **fields): pass
    def warn(self, event: str, **fields): pass
    def error(self, event: str, **fields): pass

class JsonStdoutLogger:
    def __init__(self, service: str = "agents", env: str = "dev", log_path: str | Path | None = None):
        self.service = service
        self.env = env
        self._log_path = Path(log_path).expanduser() if log_path else None
        self._log_dir_prepared = False
    def _emit(self, level: str, event: str, **fields):
        ts = (
            datetime.datetime.now(datetime.timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
        rec = {
            "ts": ts,
            "level": level,
            "event": event,
            "service": self.service,
            "env": self.env,
            **fields,
        }
        line = json.dumps(rec, default=str)
        print(line, file=sys.stdout if level != "error" else sys.stderr)
        if self._log_path:
            lock = _file_lock(self._log_path)
            with lock:
                if not self._log_dir_prepared:
                    self._log_path.parent.mkdir(parents=True, exist_ok=True)
                    self._log_dir_prepared = True
                with self._log_path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
    def info(self, event, **fields): self._emit("info", event, **fields)
    def warn(self, event, **fields): self._emit("warn", event, **fields)
    def error(self, event, **fields): self._emit("error", event, **fields)

class JsonRepoLogger(JsonStdoutLogger):
    def __init__(
        self,
        service: str = "agents",
        env: str = "dev",
        log_dir: str | Path | None = None,
        filename: str | None = None,
    ):
        override = get_config_value("OBS_LOG_FILE")
        if override and not filename:
            override_path = Path(override).expanduser()
            if not override_path.is_absolute():
                override_path = REPO_ROOT / override_path
            super().__init__(service=service, env=env, log_path=override_path)
        else:
            target_dir = Path(log_dir).expanduser() if log_dir else DEFAULT_LOG_DIR
            target_file = filename or f"{service}.log"
            super().__init__(service=service, env=env, log_path=target_dir / target_file)

def redact(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str): return value
    # super simple—tune for your needs
    return value.replace(os.getenv("OPENAI_API_KEY",""), "***") if value else value


def with_span(
    event: str,
    *,
    logger_attr: str = "_logger",
    fields: Mapping[str, Any] | None = None,
    fields_fn: Callable[..., Mapping[str, Any]] | None = None,
    pre: Callable[[tuple[Any, ...], dict[str, Any]], None] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that wraps a function in a `Span` (sync + async).

    This is a lightweight AOP-style helper for consistent `*.start`/`*.end`/`*.error`
    logs with timing.
    """

    def _resolve_logger(args: tuple[Any, ...]) -> Logger:
        if args:
            candidate = getattr(args[0], logger_attr, None)
            if candidate is not None:
                return candidate
        return NullLogger()

    def _span_fields(*args: Any, **kwargs: Any) -> dict[str, Any]:
        merged: dict[str, Any] = dict(fields or {})
        if fields_fn:
            merged.update(dict(fields_fn(*args, **kwargs)))
        return merged

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if pre:
                    pre(args, kwargs)
                logger = _resolve_logger(args)
                with Span(logger, event, _span_fields(*args, **kwargs)):
                    return await fn(*args, **kwargs)

            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if pre:
                pre(args, kwargs)
            logger = _resolve_logger(args)
            with Span(logger, event, _span_fields(*args, **kwargs)):
                return fn(*args, **kwargs)

        return sync_wrapper

    return decorator


@dataclass(slots=True)
class Span:
    logger: Logger
    event: str
    fields: Mapping[str, Any]
    start_ns: int = 0
    def __enter__(self):
        self.start_ns = time.time_ns()
        self.logger.info(self.event + ".start", **self.fields)
        return self
    def __exit__(self, exc_type, exc, tb):
        dur_ms = (time.time_ns() - self.start_ns) / 1e6
        if exc:
            self.logger.error(
                self.event + ".error",
                duration_ms=dur_ms,
                error_type=getattr(exc, "__class__", type(exc)).__name__,
                error=str(exc),
                **self.fields,
            )
        else:
            self.logger.info(self.event + ".end", duration_ms=dur_ms, **self.fields)
