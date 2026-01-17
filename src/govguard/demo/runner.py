"""Scenario runner for GovGuard demos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event as ThreadEvent, Thread
from time import sleep
from uuid import UUID, uuid4

from govguard.agents.cost_latency_agent.agent import CostLatencyAgent
from govguard.agents.drift_bias_agent.agent import DriftBiasAgent
from govguard.agents.eval_agent.agent import EvalAgent
from govguard.agents.rollback_agent.agent import RollbackAgent
from govguard.agents.security_agent.agent import SecurityAgent
from govguard.contracts.events import (
    DEPLOY_ROLLED_BACK,
    GATE_DECISION_MADE,
    MONITORING_REGRESSION_DETECTED,
    RELEASE_CANDIDATE_CREATED,
    MonitoringRegressionDetected,
    ReleaseCandidateCreated,
)
from govguard.demo.fixtures import ScenarioFixtures
from govguard.observability.logging import configure_logging
from govguard.observability.metrics import start_metrics_server
from govguard.observability.telemetry import setup_tracing
from govguard.orchestrator.event_bus import Event, InMemoryEventBus
from govguard.orchestrator.orchestrator import Orchestrator
from govguard.orchestrator.recorder import RecordingEventBus
from govguard.orchestrator.worker import AgentWorker
from govguard.registry.fixture_store import FixtureStore


@dataclass(slots=True)
class ScenarioResult:
    """Result from running a scenario."""

    candidate_id: UUID
    events: list[Event]


def run_scenario(
    fixtures: ScenarioFixtures,
    policy_path: Path,
    *,
    start_metrics: bool = True,
    enable_tracing: bool = True,
) -> ScenarioResult:
    """Run a scenario end-to-end using in-memory event bus."""
    configure_logging()
    if not enable_tracing:
        import os

        os.environ["GOVGUARD_DISABLE_TRACING"] = "1"
    setup_tracing("govguard-demo")
    if start_metrics:
        start_metrics_server()

    store = FixtureStore()
    store.register(
        fixtures.candidate.candidate_id,
        eval_metrics=fixtures.eval_metrics,
        drift_metrics=fixtures.drift_metrics,
        security_findings=fixtures.security_findings,
        cost_latency=fixtures.cost_latency,
        regression=fixtures.regression,
    )

    bus = RecordingEventBus(InMemoryEventBus())
    rollback_agent = RollbackAgent(store)
    orchestrator = Orchestrator(bus=bus, policy_path=policy_path, rollback_agent=rollback_agent)
    stop_event = ThreadEvent()

    workers = [
        AgentWorker(bus=bus, agent=EvalAgent(store)),
        AgentWorker(bus=bus, agent=DriftBiasAgent(store)),
        AgentWorker(bus=bus, agent=SecurityAgent(store)),
        AgentWorker(bus=bus, agent=CostLatencyAgent(store)),
    ]

    threads: list[Thread] = [Thread(target=orchestrator.run, args=(stop_event,), daemon=True)] + [
        Thread(target=worker.run, args=(stop_event,), daemon=True) for worker in workers
    ]

    for thread in threads:
        thread.start()

    release_payload = ReleaseCandidateCreated(candidate=fixtures.candidate)
    bus.publish(
        Event(
            event_id=uuid4(),
            event_type=RELEASE_CANDIDATE_CREATED,
            candidate_id=fixtures.candidate.candidate_id,
            payload=release_payload.model_dump(),
        )
    )

    _wait_for_event(bus.events, GATE_DECISION_MADE)

    if fixtures.regression:
        regression_payload = MonitoringRegressionDetected(
            candidate_id=fixtures.candidate.candidate_id,
            metric="accuracy",
            current_value=fixtures.regression["metric"],
            baseline_value=fixtures.regression["baseline"],
        )
        bus.publish(
            Event(
                event_id=uuid4(),
                event_type=MONITORING_REGRESSION_DETECTED,
                candidate_id=fixtures.candidate.candidate_id,
                payload=regression_payload.model_dump(),
            )
        )
        _wait_for_event(bus.events, DEPLOY_ROLLED_BACK)

    stop_event.set()
    return ScenarioResult(candidate_id=fixtures.candidate.candidate_id, events=bus.events)


def _wait_for_event(events: list[Event], event_type: str, timeout: float = 5.0) -> None:
    deadline = timeout / 0.1
    for _ in range(int(deadline)):
        if any(event.event_type == event_type for event in events):
            return
        sleep(0.1)
    raise TimeoutError(f"Timeout waiting for {event_type}")
