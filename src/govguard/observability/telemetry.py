"""OpenTelemetry tracing setup."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class _NoOpSpan:
    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        return None

    def set_attribute(self, key: str, value: Any) -> None:
        return None


class _NoOpTracer:
    def start_as_current_span(self, name: str) -> _NoOpSpan:
        return _NoOpSpan()


_NOOP_TRACER = _NoOpTracer()


def setup_tracing(service_name: str, exporter: Any | None = None) -> None:
    """Configure OpenTelemetry tracing with a console exporter."""
    if os.getenv("GOVGUARD_DISABLE_TRACING") == "1":
        logger.info("Tracing disabled", extra={"extra": {"service": service_name}})
        return
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    span_exporter = exporter or ConsoleSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(provider)
    logger.info("Tracing configured", extra={"extra": {"service": service_name}})


def get_tracer(name: str) -> Any:
    """Return a tracer instance."""
    if os.getenv("GOVGUARD_DISABLE_TRACING") == "1":
        return _NOOP_TRACER
    from opentelemetry import trace

    return trace.get_tracer(name)
