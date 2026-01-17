"""Cost and latency agent."""

from __future__ import annotations

from dataclasses import dataclass

from govguard.agents.base import AgentResult
from govguard.contracts.models import ReleaseCandidate
from govguard.registry.fixture_store import FixtureStore


@dataclass(slots=True)
class CostLatencyAgent:
    """Deterministic cost/latency agent using fixtures."""

    fixtures: FixtureStore
    name: str = "cost_latency"

    def run(self, candidate: ReleaseCandidate) -> AgentResult:
        metrics = self.fixtures.cost_latency[candidate.candidate_id]
        return AgentResult(check_name=self.name, result=metrics)
