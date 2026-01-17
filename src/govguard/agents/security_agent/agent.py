"""Security agent for PII leakage and prompt injection detection."""

from __future__ import annotations

from dataclasses import dataclass

from govguard.agents.base import AgentResult
from govguard.contracts.models import ReleaseCandidate
from govguard.registry.fixture_store import FixtureStore


@dataclass(slots=True)
class SecurityAgent:
    """Deterministic security agent using fixtures."""

    fixtures: FixtureStore
    name: str = "security"

    def run(self, candidate: ReleaseCandidate) -> AgentResult:
        findings = self.fixtures.security_findings[candidate.candidate_id]
        return AgentResult(check_name=self.name, result=findings)
