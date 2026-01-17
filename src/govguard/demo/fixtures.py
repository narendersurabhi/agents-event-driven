"""Fixture data for demo scenarios."""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import UUID, uuid4

from govguard.contracts.models import (
    ArtifactRef,
    CostLatencyMetrics,
    DriftBiasMetrics,
    EvalMetrics,
    Lineage,
    ReleaseCandidate,
    SecurityFindings,
)
from govguard.contracts.types import ArtifactType, Environment


class ScenarioFixtures:
    """Container for fixture data for a scenario."""

    def __init__(
        self,
        candidate: ReleaseCandidate,
        eval_metrics: EvalMetrics,
        drift_metrics: DriftBiasMetrics,
        security_findings: SecurityFindings,
        cost_latency: CostLatencyMetrics,
        regression: dict[str, float] | None = None,
    ) -> None:
        self.candidate = candidate
        self.eval_metrics = eval_metrics
        self.drift_metrics = drift_metrics
        self.security_findings = security_findings
        self.cost_latency = cost_latency
        self.regression = regression


def _candidate(candidate_id: UUID, env: Environment, risk_tier: int) -> ReleaseCandidate:
    return ReleaseCandidate(
        candidate_id=candidate_id,
        artifact_refs=[
            ArtifactRef(type=ArtifactType.MODEL, version="1.3.0", digest="sha256:abc"),
            ArtifactRef(type=ArtifactType.PROMPT, version="2.1.0", digest="sha256:def"),
        ],
        baseline_ref=ArtifactRef(type=ArtifactType.MODEL, version="1.2.5", digest="sha256:base"),
        env=env,
        risk_tier=risk_tier,
        lineage=Lineage(
            code_sha="deadbeef",
            data_snapshot_id="snapshot-2025-01-17",
            feature_schema_version="v7",
            build_timestamp=datetime.utcnow() - timedelta(hours=2),
        ),
    )


def happy_path(candidate_id: UUID | None = None) -> ScenarioFixtures:
    cid = candidate_id or uuid4()
    return ScenarioFixtures(
        candidate=_candidate(cid, Environment.PROD, 1),
        eval_metrics=EvalMetrics(
            auroc=0.86, f1_score=0.81, groundedness=0.87, citation_coverage=0.92
        ),
        drift_metrics=DriftBiasMetrics(psi=0.11, slice_deltas={"region:emea": 0.03}),
        security_findings=SecurityFindings(
            pii_leak_detected=False,
            prompt_injection_detected=False,
            findings=[],
        ),
        cost_latency=CostLatencyMetrics(
            p50_ms=220,
            p95_ms=640,
            tokens_per_request=780,
            cost_usd_per_1k=0.009,
        ),
    )


def blocked_path(candidate_id: UUID | None = None) -> ScenarioFixtures:
    cid = candidate_id or uuid4()
    return ScenarioFixtures(
        candidate=_candidate(cid, Environment.PROD, 2),
        eval_metrics=EvalMetrics(
            auroc=0.83, f1_score=0.79, groundedness=0.84, citation_coverage=0.9
        ),
        drift_metrics=DriftBiasMetrics(psi=0.12, slice_deltas={"segment:high_risk": 0.02}),
        security_findings=SecurityFindings(
            pii_leak_detected=True,
            prompt_injection_detected=True,
            findings=["SSN pattern exposed", "Prompt injection: ignore previous"],
        ),
        cost_latency=CostLatencyMetrics(
            p50_ms=240,
            p95_ms=610,
            tokens_per_request=820,
            cost_usd_per_1k=0.009,
        ),
    )


def rollback_path(candidate_id: UUID | None = None) -> ScenarioFixtures:
    cid = candidate_id or uuid4()
    return ScenarioFixtures(
        candidate=_candidate(cid, Environment.STAGE, 1),
        eval_metrics=EvalMetrics(
            auroc=0.81, f1_score=0.76, groundedness=0.82, citation_coverage=0.88
        ),
        drift_metrics=DriftBiasMetrics(psi=0.14, slice_deltas={"segment:new": 0.06}),
        security_findings=SecurityFindings(
            pii_leak_detected=False,
            prompt_injection_detected=False,
            findings=[],
        ),
        cost_latency=CostLatencyMetrics(
            p50_ms=260,
            p95_ms=720,
            tokens_per_request=900,
            cost_usd_per_1k=0.012,
        ),
        regression={"metric": 0.42, "baseline": 0.12},
    )
