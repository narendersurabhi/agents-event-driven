"""Integration tests for GovGuard scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from govguard.contracts.events import DEPLOY_ROLLED_BACK, GATE_DECISION_MADE
from govguard.contracts.types import GateDecisionType
from govguard.demo import fixtures
from govguard.demo.runner import run_scenario
from govguard.orchestrator.event_bus import Event


def _decision_for(events: list[Event]) -> str | None:
    for event in events:
        if event.event_type == GATE_DECISION_MADE:
            payload = cast(dict[str, dict[str, str]], event.payload)
            return payload["decision"]["decision"]
    return None


def test_happy_path_gate_approves() -> None:
    scenario = fixtures.happy_path()
    result = run_scenario(
        scenario,
        Path("src/govguard/gatekeeper/policy.yaml"),
        start_metrics=False,
        enable_tracing=False,
    )
    assert _decision_for(result.events) == GateDecisionType.APPROVE.value


def test_blocked_path_gate_blocks() -> None:
    scenario = fixtures.blocked_path()
    result = run_scenario(
        scenario,
        Path("src/govguard/gatekeeper/policy.yaml"),
        start_metrics=False,
        enable_tracing=False,
    )
    assert _decision_for(result.events) == GateDecisionType.BLOCK.value


def test_rollback_path_triggers_rollback() -> None:
    scenario = fixtures.rollback_path()
    result = run_scenario(
        scenario,
        Path("src/govguard/gatekeeper/policy.yaml"),
        start_metrics=False,
        enable_tracing=False,
    )
    assert any(event.event_type == DEPLOY_ROLLED_BACK for event in result.events)
