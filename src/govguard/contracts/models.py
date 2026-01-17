"""Domain models for model governance and release gating."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from govguard.contracts.types import ArtifactType, Environment, GateDecisionType


class ArtifactRef(BaseModel):
    """Reference to a governed artifact."""

    type: ArtifactType
    version: str
    digest: str


class Lineage(BaseModel):
    """Lineage metadata for a release candidate."""

    code_sha: str
    data_snapshot_id: str
    feature_schema_version: str
    build_timestamp: datetime


class ReleaseCandidate(BaseModel):
    """Candidate artifact bundle for evaluation and release gating."""

    candidate_id: UUID
    artifact_refs: list[ArtifactRef]
    baseline_ref: ArtifactRef | None = None
    env: Environment
    risk_tier: int = Field(ge=0, le=2)
    lineage: Lineage


class EvalMetrics(BaseModel):
    """Metrics for evaluation agents."""

    auroc: float | None = None
    f1_score: float | None = None
    groundedness: float | None = None
    citation_coverage: float | None = None
    hallucination_flag: bool = False
    notes: list[str] = Field(default_factory=list)


class DriftBiasMetrics(BaseModel):
    """Metrics for drift and bias detection."""

    psi: float
    slice_deltas: dict[str, float]


class SecurityFindings(BaseModel):
    """Security scan findings."""

    pii_leak_detected: bool
    prompt_injection_detected: bool
    findings: list[str] = Field(default_factory=list)


class CostLatencyMetrics(BaseModel):
    """Cost and latency metrics."""

    p50_ms: float
    p95_ms: float
    tokens_per_request: float
    cost_usd_per_1k: float


class GateDecision(BaseModel):
    """Gate decision for a candidate."""

    candidate_id: UUID
    decision: GateDecisionType
    reasons: list[str]
    required_checks_passed: bool
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeployRecord(BaseModel):
    """Deployment status record."""

    candidate_id: UUID
    approved: bool
    started_at: datetime
    completed_at: datetime | None = None
    rolled_back: bool = False
