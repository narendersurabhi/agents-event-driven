"""Tests for drift metrics utilities."""

from __future__ import annotations

from govguard.agents.drift_bias_agent.metrics import calculate_psi


def test_calculate_psi_positive() -> None:
    expected = [0.2, 0.3, 0.5]
    actual = [0.25, 0.25, 0.5]
    psi = calculate_psi(expected, actual)
    assert psi > 0
