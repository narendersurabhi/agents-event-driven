"""Evaluation agent for classic ML + LLM metrics."""

from __future__ import annotations

from dataclasses import dataclass

from govguard.agents.base import AgentResult
from govguard.contracts.models import ReleaseCandidate
from govguard.registry.fixture_store import FixtureStore


@dataclass(slots=True)
class EvalAgent:
    """Deterministic evaluation agent using fixtures."""

    fixtures: FixtureStore
    name: str = "eval"

    def run(self, candidate: ReleaseCandidate) -> AgentResult:
        metrics = self.fixtures.eval_metrics[candidate.candidate_id]
        return AgentResult(check_name=self.name, result=metrics)
