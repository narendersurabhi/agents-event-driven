"""Long-running GovGuard service for Docker Compose."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Event as ThreadEvent, Thread

from govguard.agents.cost_latency_agent.agent import CostLatencyAgent
from govguard.agents.drift_bias_agent.agent import DriftBiasAgent
from govguard.agents.eval_agent.agent import EvalAgent
from govguard.agents.rollback_agent.agent import RollbackAgent
from govguard.agents.security_agent.agent import SecurityAgent
from govguard.demo.fixtures import happy_path
from govguard.observability.logging import configure_logging
from govguard.observability.metrics import start_metrics_server
from govguard.observability.telemetry import setup_tracing
from govguard.orchestrator.event_bus import EventBus, InMemoryEventBus
from govguard.orchestrator.nats_bus import NATSEventBus
from govguard.orchestrator.orchestrator import Orchestrator
from govguard.orchestrator.worker import AgentWorker
from govguard.registry.fixture_store import FixtureStore


def main() -> None:
    configure_logging()
    setup_tracing("govguard-service")
    start_metrics_server(port=int(os.getenv("GOVGUARD_METRICS_PORT", "8005")))

    bus_type = os.getenv("GOVGUARD_BUS", "memory")
    if bus_type == "nats":
        bus: EventBus = NATSEventBus(os.getenv("NATS_URL", "nats://nats:4222"))
    else:
        bus = InMemoryEventBus()

    fixtures = happy_path()
    store = FixtureStore()
    store.register(
        fixtures.candidate.candidate_id,
        eval_metrics=fixtures.eval_metrics,
        drift_metrics=fixtures.drift_metrics,
        security_findings=fixtures.security_findings,
        cost_latency=fixtures.cost_latency,
    )

    rollback_agent = RollbackAgent(store)
    orchestrator = Orchestrator(
        bus=bus,
        policy_path=Path("src/govguard/gatekeeper/policy.yaml"),
        rollback_agent=rollback_agent,
    )

    workers = [
        AgentWorker(bus=bus, agent=EvalAgent(store)),
        AgentWorker(bus=bus, agent=DriftBiasAgent(store)),
        AgentWorker(bus=bus, agent=SecurityAgent(store)),
        AgentWorker(bus=bus, agent=CostLatencyAgent(store)),
    ]

    stop_event = ThreadEvent()
    threads = [Thread(target=orchestrator.run, args=(stop_event,), daemon=True)] + [
        Thread(target=worker.run, args=(stop_event,), daemon=True) for worker in workers
    ]

    for thread in threads:
        thread.start()

    stop_event.wait()


if __name__ == "__main__":
    main()
