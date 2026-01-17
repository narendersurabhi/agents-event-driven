"""Release workflow orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from threading import Event as ThreadEvent
from typing import Any
from uuid import UUID, uuid4

from govguard.agents.rollback_agent.agent import RollbackAgent
from govguard.contracts.events import (
    DEPLOY_APPROVED,
    DEPLOY_BLOCKED,
    DEPLOY_COMPLETED,
    DEPLOY_ROLLED_BACK,
    DEPLOY_STARTED,
    EVAL_COMPLETED,
    EVAL_FAILED,
    EVAL_STARTED,
    EVAL_WARNING,
    GATE_DECISION_MADE,
    MONITORING_REGRESSION_DETECTED,
    RELEASE_CANDIDATE_CREATED,
    DeployApproved,
    DeployBlocked,
    DeployCompleted,
    DeployRolledBack,
    DeployStarted,
    EvalCompleted,
    EvalFailed,
    EvalStarted,
    EvalWarning,
    GateDecisionMade,
    MonitoringRegressionDetected,
    ReleaseCandidateCreated,
)
from govguard.contracts.models import GateDecision, ReleaseCandidate
from govguard.gatekeeper.policy import Gatekeeper, PolicyConfig
from govguard.observability.metrics import DECISIONS
from govguard.observability.telemetry import get_tracer
from govguard.orchestrator.event_bus import Event, EventBus
from govguard.orchestrator.state import OrchestratorState

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Orchestrator:
    """Coordinates the release candidate lifecycle."""

    bus: EventBus
    gatekeeper: Gatekeeper
    state: OrchestratorState
    rollback_agent: RollbackAgent | None = None

    def __init__(
        self, bus: EventBus, policy_path: Path, rollback_agent: RollbackAgent | None = None
    ) -> None:
        self.bus = bus
        self.gatekeeper = Gatekeeper(PolicyConfig.load(policy_path))
        self.state = OrchestratorState()
        self.rollback_agent = rollback_agent

    def run(self, stop_event: ThreadEvent) -> None:
        """Main loop for orchestrator events."""
        tracer = get_tracer("govguard.orchestrator")
        while not stop_event.is_set():
            event = self.bus.next_event(RELEASE_CANDIDATE_CREATED, timeout=0.2)
            if event:
                self._handle_release_created(event, tracer)
            event = self.bus.next_event(EVAL_COMPLETED, timeout=0.1)
            if event:
                self._handle_eval_completed(event, tracer)
            event = self.bus.next_event(EVAL_FAILED, timeout=0.1)
            if event:
                self._handle_eval_failed(event, tracer)
            event = self.bus.next_event(EVAL_WARNING, timeout=0.1)
            if event:
                self._handle_eval_warning(event, tracer)
            event = self.bus.next_event(MONITORING_REGRESSION_DETECTED, timeout=0.1)
            if event:
                self._handle_regression(event, tracer)

    def _handle_release_created(self, event: Event, tracer: Any) -> None:
        if event.event_id in self.state.processed_event_ids:
            return
        with tracer.start_as_current_span("release.candidate.created") as span:
            payload = ReleaseCandidateCreated.model_validate(event.payload)
            state = self.state.get_or_create(payload.candidate)
            span.set_attribute("candidate_id", str(payload.candidate.candidate_id))
            logger.info(
                "release.candidate.created",
                extra={"extra": {"candidate_id": str(payload.candidate.candidate_id)}},
            )
            self.state.processed_event_ids.add(event.event_id)
            for check in ("eval", "drift_bias", "security", "cost_latency"):
                eval_payload = EvalStarted(candidate=payload.candidate, check_name=check)
                self._publish(
                    EVAL_STARTED,
                    payload.candidate.candidate_id,
                    eval_payload.model_dump(),
                )
            state.deploy_started_at = None

    def _handle_eval_completed(self, event: Event, tracer: Any) -> None:
        with tracer.start_as_current_span("eval.completed") as span:
            payload = EvalCompleted.model_validate(event.payload)
            state = self.state.candidates[payload.candidate_id]
            state.eval_results[payload.check_name] = payload.metrics
            span.set_attribute("candidate_id", str(payload.candidate_id))
            span.set_attribute("check_name", payload.check_name)
            logger.info(
                "eval.completed",
                extra={
                    "extra": {
                        "candidate_id": str(payload.candidate_id),
                        "check": payload.check_name,
                    }
                },
            )
            if self._all_checks_complete(state.candidate, state.eval_results):
                decision = self.gatekeeper.decide(state.candidate, state.eval_results)
                state.gate_decision = decision
                self._publish(
                    GATE_DECISION_MADE,
                    payload.candidate_id,
                    GateDecisionMade(decision=decision).model_dump(),
                )
                self._handle_gate_decision(decision)

    def _handle_eval_failed(self, event: Event, tracer: Any) -> None:
        with tracer.start_as_current_span("eval.failed") as span:
            payload = EvalFailed.model_validate(event.payload)
            span.set_attribute("candidate_id", str(payload.candidate_id))
            span.set_attribute("check_name", payload.check_name)
            logger.error(
                "eval.failed",
                extra={
                    "extra": {
                        "candidate_id": str(payload.candidate_id),
                        "check": payload.check_name,
                        "reason": payload.reason,
                    }
                },
            )

    def _handle_eval_warning(self, event: Event, tracer: Any) -> None:
        with tracer.start_as_current_span("eval.warning") as span:
            payload = EvalWarning.model_validate(event.payload)
            span.set_attribute("candidate_id", str(payload.candidate_id))
            span.set_attribute("check_name", payload.check_name)
            logger.warning(
                "eval.warning",
                extra={
                    "extra": {
                        "candidate_id": str(payload.candidate_id),
                        "check": payload.check_name,
                        "warning": payload.warning,
                    }
                },
            )

    def _handle_gate_decision(self, decision: GateDecision) -> None:
        DECISIONS.labels(decision=decision.decision.value).inc()
        if decision.decision.value == "BLOCK":
            payload = DeployBlocked(candidate_id=decision.candidate_id, reasons=decision.reasons)
            self._publish(DEPLOY_BLOCKED, decision.candidate_id, payload.model_dump())
            return
        approve_payload = DeployApproved(candidate_id=decision.candidate_id)
        self._publish(DEPLOY_APPROVED, decision.candidate_id, approve_payload.model_dump())
        self._publish(
            DEPLOY_STARTED,
            decision.candidate_id,
            DeployStarted(candidate_id=decision.candidate_id).model_dump(),
        )
        complete_payload = DeployCompleted(candidate_id=decision.candidate_id)
        self._publish(DEPLOY_COMPLETED, decision.candidate_id, complete_payload.model_dump())

    def _handle_regression(self, event: Event, tracer: Any) -> None:
        with tracer.start_as_current_span("monitoring.regression.detected") as span:
            payload = MonitoringRegressionDetected.model_validate(event.payload)
            span.set_attribute("candidate_id", str(payload.candidate_id))
            logger.warning(
                "monitoring.regression.detected",
                extra={
                    "extra": {
                        "candidate_id": str(payload.candidate_id),
                        "metric": payload.metric,
                        "current": payload.current_value,
                        "baseline": payload.baseline_value,
                    }
                },
            )
            should_rollback = True
            if self.rollback_agent:
                should_rollback = self.rollback_agent.should_rollback(payload)
            if should_rollback:
                rollback_payload = DeployRolledBack(
                    candidate_id=payload.candidate_id,
                    reason=f"Regression detected for {payload.metric}",
                )
                self._publish(
                    DEPLOY_ROLLED_BACK, payload.candidate_id, rollback_payload.model_dump()
                )

    def _publish(self, event_type: str, candidate_id: UUID, payload: dict[str, Any]) -> None:
        event = Event(
            event_id=uuid4(),
            event_type=event_type,
            candidate_id=candidate_id,
            payload=payload,
        )
        self.bus.publish(event)

    @staticmethod
    def _all_checks_complete(candidate: ReleaseCandidate, results: dict[str, Any]) -> bool:
        required = {"eval", "drift_bias", "security", "cost_latency"}
        return required.issubset(results.keys())
