"""State tracking for the release workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from govguard.contracts.models import GateDecision, ReleaseCandidate


@dataclass(slots=True)
class CandidateState:
    """Mutable state for a release candidate during orchestration."""

    candidate: ReleaseCandidate
    eval_results: dict[str, Any] = field(default_factory=dict)
    gate_decision: GateDecision | None = None
    deploy_started_at: datetime | None = None
    deploy_completed_at: datetime | None = None
    rolled_back: bool = False


@dataclass(slots=True)
class OrchestratorState:
    """Central store for candidate states."""

    candidates: dict[UUID, CandidateState] = field(default_factory=dict)
    processed_event_ids: set[UUID] = field(default_factory=set)

    def get_or_create(self, candidate: ReleaseCandidate) -> CandidateState:
        if candidate.candidate_id not in self.candidates:
            self.candidates[candidate.candidate_id] = CandidateState(candidate=candidate)
        return self.candidates[candidate.candidate_id]
