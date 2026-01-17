"""Rollback decision agent."""

from __future__ import annotations

from dataclasses import dataclass

from govguard.contracts.events import MonitoringRegressionDetected
from govguard.registry.fixture_store import FixtureStore


@dataclass(slots=True)
class RollbackAgent:
    """Determines whether to rollback based on regression metrics."""

    fixtures: FixtureStore
    threshold: float = 0.2

    def should_rollback(self, event: MonitoringRegressionDetected) -> bool:
        if event.candidate_id not in self.fixtures.regression_metrics:
            return False
        regression = self.fixtures.regression_metrics[event.candidate_id]
        return regression.get("metric", 0.0) - regression.get("baseline", 0.0) > self.threshold
