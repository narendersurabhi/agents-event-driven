"""Drift and bias detection agent."""

from __future__ import annotations

from dataclasses import dataclass

from govguard.agents.base import AgentResult
from govguard.contracts.models import ReleaseCandidate
from govguard.registry.fixture_store import FixtureStore


@dataclass(slots=True)
class DriftBiasAgent:
    """Deterministic drift/bias agent using fixtures."""

    fixtures: FixtureStore
    name: str = "drift_bias"

    def run(self, candidate: ReleaseCandidate) -> AgentResult:
        metrics = self.fixtures.drift_metrics[candidate.candidate_id]
        return AgentResult(check_name=self.name, result=metrics)
