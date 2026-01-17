"""Agent worker that listens for eval events."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from threading import Event as ThreadEvent
from time import sleep
from uuid import UUID, uuid4

from govguard.agents.base import Agent
from govguard.contracts.events import (
    EVAL_COMPLETED,
    EVAL_FAILED,
    EVAL_STARTED,
    EvalCompleted,
    EvalFailed,
    EvalStarted,
)
from govguard.observability.metrics import EVAL_DURATION
from govguard.observability.telemetry import get_tracer
from govguard.orchestrator.event_bus import Event, EventBus

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AgentWorker:
    """Worker that executes an agent when an eval starts."""

    bus: EventBus
    agent: Agent

    def run(self, stop_event: ThreadEvent) -> None:
        tracer = get_tracer(f"govguard.agent.{self.agent.name}")
        while not stop_event.is_set():
            event = self.bus.next_event(EVAL_STARTED, timeout=0.2)
            if not event:
                continue
            payload = EvalStarted.model_validate(event.payload)
            if payload.check_name != self.agent.name:
                self.bus.publish(event)
                sleep(0.01)
                continue
            with tracer.start_as_current_span(f"agent.{self.agent.name}.run"):
                try:
                    with EVAL_DURATION.labels(check_name=self.agent.name).time():
                        result = self.agent.run(payload.candidate)
                    completed = EvalCompleted(
                        candidate_id=payload.candidate.candidate_id,
                        check_name=result.check_name,
                        metrics=result.result,
                    )
                    self._publish(
                        EVAL_COMPLETED, payload.candidate.candidate_id, completed.model_dump()
                    )
                except Exception as exc:  # noqa: BLE001
                    failed = EvalFailed(
                        candidate_id=payload.candidate.candidate_id,
                        check_name=self.agent.name,
                        reason=str(exc),
                    )
                    logger.exception("agent.failed", extra={"extra": {"agent": self.agent.name}})
                    self._publish(EVAL_FAILED, payload.candidate.candidate_id, failed.model_dump())

    def _publish(self, event_type: str, candidate_id: UUID, payload: dict[str, object]) -> None:
        event = Event(
            event_id=uuid4(),
            event_type=event_type,
            candidate_id=candidate_id,
            payload=payload,
        )
        self.bus.publish(event)
