"""Unit tests for the gatekeeper policy engine."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from govguard.contracts.models import (
    ArtifactRef,
    CostLatencyMetrics,
    DriftBiasMetrics,
    EvalMetrics,
    Lineage,
    ReleaseCandidate,
    SecurityFindings,
)
from govguard.contracts.types import ArtifactType, Environment, GateDecisionType
from govguard.gatekeeper.policy import Gatekeeper, PolicyConfig


def _candidate(env: Environment, risk_tier: int) -> ReleaseCandidate:
    return ReleaseCandidate(
        candidate_id=uuid4(),
        artifact_refs=[ArtifactRef(type=ArtifactType.MODEL, version="1.0", digest="sha256:abc")],
        baseline_ref=None,
        env=env,
        risk_tier=risk_tier,
        lineage=Lineage(
            code_sha="abc",
            data_snapshot_id="snap",
            feature_schema_version="v1",
            build_timestamp=datetime.utcnow(),
        ),
    )


def test_gatekeeper_blocks_on_security_findings() -> None:
    policy = PolicyConfig.load(Path("src/govguard/gatekeeper/policy.yaml"))
    gatekeeper = Gatekeeper(policy)
    candidate = _candidate(Environment.PROD, 1)
    results = {
        "eval": EvalMetrics(auroc=0.85, f1_score=0.78, groundedness=0.9, citation_coverage=0.92),
        "drift_bias": DriftBiasMetrics(psi=0.1, slice_deltas={"segment": 0.02}),
        "security": SecurityFindings(
            pii_leak_detected=True,
            prompt_injection_detected=False,
            findings=["PII leak"],
        ),
        "cost_latency": CostLatencyMetrics(
            p50_ms=200,
            p95_ms=500,
            tokens_per_request=700,
            cost_usd_per_1k=0.009,
        ),
    }
    decision = gatekeeper.decide(candidate, results)
    assert decision.decision == GateDecisionType.BLOCK


def test_gatekeeper_approves_with_warnings_in_dev() -> None:
    policy = PolicyConfig.load(Path("src/govguard/gatekeeper/policy.yaml"))
    gatekeeper = Gatekeeper(policy)
    candidate = _candidate(Environment.DEV, 0)
    results = {
        "eval": EvalMetrics(auroc=0.8, f1_score=0.75, groundedness=0.6, citation_coverage=0.7),
        "drift_bias": DriftBiasMetrics(psi=0.1, slice_deltas={"segment": 0.02}),
        "security": SecurityFindings(
            pii_leak_detected=False,
            prompt_injection_detected=False,
            findings=[],
        ),
        "cost_latency": CostLatencyMetrics(
            p50_ms=200,
            p95_ms=500,
            tokens_per_request=700,
            cost_usd_per_1k=0.03,
        ),
    }
    decision = gatekeeper.decide(candidate, results)
    assert decision.decision == GateDecisionType.APPROVE_WITH_WARNINGS
