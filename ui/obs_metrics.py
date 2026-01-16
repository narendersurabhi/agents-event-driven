from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, cast


def read_recent_jsonl(text: str, *, max_lines: int = 5000) -> list[dict[str, Any]]:
    lines = text.splitlines()[-max_lines:]
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            records.append(rec)
    return records


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values_sorted = sorted(values)
    idx = int(round((len(values_sorted) - 1) * p))
    idx = max(0, min(idx, len(values_sorted) - 1))
    return float(values_sorted[idx])


def fmt_ms(value: float | None) -> str:
    return f"{value:.1f}" if isinstance(value, (int, float)) else ""


@dataclass(slots=True)
class StepAgg:
    durations_ms: list[float] = field(default_factory=list)
    error_durations_ms: list[float] = field(default_factory=list)
    success: int = 0
    errors: int = 0
    repairs: int = 0
    parse_failures: int = 0
    last_ts: str = ""


def aggregate_step_metrics(records: list[dict[str, Any]]) -> dict[str, StepAgg]:
    stats: dict[str, StepAgg] = {}

    def ensure(step: str) -> StepAgg:
        if step not in stats:
            stats[step] = StepAgg()
        return stats[step]

    for rec in records:
        event = rec.get("event", "")
        step = rec.get("step", "unknown") or "unknown"
        ts = rec.get("ts", "") or ""

        if event == "llm_step.end":
            agg = ensure(step)
            agg.success += 1
            dur = rec.get("duration_ms")
            if isinstance(dur, (int, float)):
                agg.durations_ms.append(float(dur))
            if bool(rec.get("repaired", False)):
                agg.repairs += 1
            if ts:
                agg.last_ts = max(agg.last_ts, ts)
            continue

        if event == "llm_step.error":
            agg = ensure(step)
            agg.errors += 1
            dur = rec.get("duration_ms")
            if isinstance(dur, (int, float)):
                agg.error_durations_ms.append(float(dur))
            if ts:
                agg.last_ts = max(agg.last_ts, ts)
            continue

        if event == "llm_step.parse_failed":
            agg = ensure(step)
            agg.parse_failures += 1
            if ts:
                agg.last_ts = max(agg.last_ts, ts)

    return stats


def build_step_rows(stats: dict[str, StepAgg]) -> list[dict[str, object]]:
    order = {
        "jd": 1,
        "profile": 2,
        "match": 3,
        "compose": 4,
        "qa": 5,
        "qa_improve": 6,
        "cover_letter": 7,
    }

    rows: list[dict[str, object]] = []
    for step, agg in stats.items():
        calls = agg.success + agg.errors
        p50 = percentile(agg.durations_ms, 0.50)
        p95 = percentile(agg.durations_ms, 0.95)
        avg = (sum(agg.durations_ms) / len(agg.durations_ms)) if agg.durations_ms else None
        repair_rate = (agg.repairs / agg.success * 100.0) if agg.success else 0.0

        rows.append(
            {
                "task": step,
                "calls": calls,
                "success": agg.success,
                "errors": agg.errors,
                "p50_ms": fmt_ms(p50),
                "p95_ms": fmt_ms(p95),
                "avg_ms": fmt_ms(avg),
                "repair_%": f"{repair_rate:.0f}",
                "parse_failures": agg.parse_failures,
                "last_ts": agg.last_ts,
            }
        )

    rows.sort(key=lambda r: (order.get(str(r["task"]), 999), -int(cast(int, r["calls"]))))
    return rows
