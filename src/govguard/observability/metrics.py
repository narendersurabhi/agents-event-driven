"""Prometheus-style metrics utilities without external dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from types import TracebackType


@dataclass
class _LabeledCounter:
    value: float = 0.0

    def inc(self, amount: float = 1.0) -> None:
        self.value += amount


@dataclass
class _LabeledHistogram:
    count: int = 0
    total: float = 0.0

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value

    def time(self) -> _Timer:
        return _Timer(self)


@dataclass
class Counter:
    name: str
    description: str
    label_names: tuple[str, ...]
    values: dict[tuple[str, ...], _LabeledCounter] = field(default_factory=dict)

    def labels(self, **labels: str) -> _LabeledCounter:
        key = tuple(labels[name] for name in self.label_names)
        if key not in self.values:
            self.values[key] = _LabeledCounter()
        return self.values[key]


@dataclass
class Histogram:
    name: str
    description: str
    label_names: tuple[str, ...]
    values: dict[tuple[str, ...], _LabeledHistogram] = field(default_factory=dict)

    def labels(self, **labels: str) -> _LabeledHistogram:
        key = tuple(labels[name] for name in self.label_names)
        if key not in self.values:
            self.values[key] = _LabeledHistogram()
        return self.values[key]


DECISIONS = Counter(
    name="govguard_gate_decisions_total",
    description="Count of gate decisions by outcome",
    label_names=("decision",),
)

EVAL_DURATION = Histogram(
    name="govguard_eval_duration_seconds",
    description="Duration of evaluation checks",
    label_names=("check_name",),
)


class _MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        body = _render_metrics().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


_server_thread: Thread | None = None


def start_metrics_server(port: int = 8005) -> None:
    """Start a lightweight Prometheus-style metrics server."""
    global _server_thread
    if _server_thread:
        return

    server = HTTPServer(("0.0.0.0", port), _MetricsHandler)

    def _run() -> None:
        server.serve_forever()

    _server_thread = Thread(target=_run, daemon=True)
    _server_thread.start()


def _render_metrics() -> str:
    lines: list[str] = []
    lines.append(f"# HELP {DECISIONS.name} {DECISIONS.description}")
    lines.append(f"# TYPE {DECISIONS.name} counter")
    for labels, counter in DECISIONS.values.items():
        label_str = f'decision="{labels[0]}"'
        lines.append(f"{DECISIONS.name}{{{label_str}}} {counter.value}")

    lines.append(f"# HELP {EVAL_DURATION.name} {EVAL_DURATION.description}")
    lines.append(f"# TYPE {EVAL_DURATION.name} summary")
    for labels, histogram in EVAL_DURATION.values.items():
        label_str = f'check_name="{labels[0]}"'
        lines.append(f"{EVAL_DURATION.name}_count{{{label_str}}} {histogram.count}")
        lines.append(f"{EVAL_DURATION.name}_sum{{{label_str}}} {histogram.total}")

    return "\n".join(lines) + "\n"


class _Timer:
    def __init__(self, histogram: _LabeledHistogram) -> None:
        self._histogram = histogram
        self._start: float | None = None

    def __enter__(self) -> _Timer:
        import time

        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        import time

        if self._start is None:
            return
        duration = time.perf_counter() - self._start
        self._histogram.observe(duration)
