"""Drift and bias metrics utilities."""

from __future__ import annotations

from math import log


def calculate_psi(expected: list[float], actual: list[float]) -> float:
    """Calculate Population Stability Index (PSI)."""
    if len(expected) != len(actual):
        raise ValueError("expected and actual distributions must be the same length")
    psi = 0.0
    for exp, act in zip(expected, actual, strict=True):
        if exp <= 0 or act <= 0:
            raise ValueError("distributions must contain positive values")
        psi += (act - exp) * log(act / exp)
    return psi
