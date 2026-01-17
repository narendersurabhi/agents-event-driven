"""Event contracts and helpers for GovGuard."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from govguard.contracts.models import (
    CostLatencyMetrics,
    DriftBiasMetrics,
    EvalMetrics,
    GateDecision,
    ReleaseCandidate,
    SecurityFindings,
)


class EventEnvelope(BaseModel):
    """Standard envelope for all events."""

    event_id: UUID = Field(default_factory=uuid4)
    event_type: str
    candidate_id: UUID
    correlation_id: UUID | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    payload: dict[str, Any]


class ReleaseCandidateCreated(BaseModel):
    candidate: ReleaseCandidate


class EvalStarted(BaseModel):
    candidate: ReleaseCandidate
    check_name: str


class EvalCompleted(BaseModel):
    candidate_id: UUID
    check_name: str
    metrics: EvalMetrics | DriftBiasMetrics | SecurityFindings | CostLatencyMetrics


class EvalFailed(BaseModel):
    candidate_id: UUID
    check_name: str
    reason: str


class EvalWarning(BaseModel):
    candidate_id: UUID
    check_name: str
    warning: str


class GateDecisionMade(BaseModel):
    decision: GateDecision


class DeployStarted(BaseModel):
    candidate_id: UUID


class DeployApproved(BaseModel):
    candidate_id: UUID


class DeployBlocked(BaseModel):
    candidate_id: UUID
    reasons: list[str]


class DeployCompleted(BaseModel):
    candidate_id: UUID


class MonitoringRegressionDetected(BaseModel):
    candidate_id: UUID
    metric: str
    current_value: float
    baseline_value: float


class DeployRolledBack(BaseModel):
    candidate_id: UUID
    reason: str


RELEASE_CANDIDATE_CREATED = "release.candidate.created"
EVAL_STARTED = "eval.started"
EVAL_COMPLETED = "eval.completed"
EVAL_FAILED = "eval.failed"
EVAL_WARNING = "eval.warning"
GATE_DECISION_MADE = "gate.decision.made"
DEPLOY_STARTED = "deploy.started"
DEPLOY_APPROVED = "deploy.approved"
DEPLOY_BLOCKED = "deploy.blocked"
DEPLOY_COMPLETED = "deploy.completed"
MONITORING_REGRESSION_DETECTED = "monitoring.regression.detected"
DEPLOY_ROLLED_BACK = "deploy.rolled_back"
