"""Fixture store for deterministic agent outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from govguard.contracts.models import (
    CostLatencyMetrics,
    DriftBiasMetrics,
    EvalMetrics,
    SecurityFindings,
)


@dataclass(slots=True)
class FixtureStore:
    """In-memory lookup for agent fixtures keyed by candidate."""

    eval_metrics: dict[UUID, EvalMetrics] = field(default_factory=dict)
    drift_metrics: dict[UUID, DriftBiasMetrics] = field(default_factory=dict)
    security_findings: dict[UUID, SecurityFindings] = field(default_factory=dict)
    cost_latency: dict[UUID, CostLatencyMetrics] = field(default_factory=dict)
    regression_metrics: dict[UUID, dict[str, float]] = field(default_factory=dict)

    def register(
        self,
        candidate_id: UUID,
        *,
        eval_metrics: EvalMetrics,
        drift_metrics: DriftBiasMetrics,
        security_findings: SecurityFindings,
        cost_latency: CostLatencyMetrics,
        regression: dict[str, float] | None = None,
    ) -> None:
        self.eval_metrics[candidate_id] = eval_metrics
        self.drift_metrics[candidate_id] = drift_metrics
        self.security_findings[candidate_id] = security_findings
        self.cost_latency[candidate_id] = cost_latency
        if regression:
            self.regression_metrics[candidate_id] = regression
