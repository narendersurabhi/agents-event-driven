"""Agent interfaces for GovGuard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from govguard.contracts.models import (
    CostLatencyMetrics,
    DriftBiasMetrics,
    EvalMetrics,
    ReleaseCandidate,
    SecurityFindings,
)


@dataclass(slots=True)
class AgentResult:
    """Container for agent results."""

    check_name: str
    result: EvalMetrics | DriftBiasMetrics | SecurityFindings | CostLatencyMetrics


class Agent(Protocol):
    """Protocol for deterministic agent checks."""

    name: str

    def run(self, candidate: ReleaseCandidate) -> AgentResult:
        """Run agent evaluation for a candidate."""
