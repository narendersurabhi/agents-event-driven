"""Backward-compatible import path for JSON repair.

The implementation lives in `core/json_repair.py` to keep infrastructure in `core`.
"""

from __future__ import annotations

from core.json_repair import JsonRepairAgent

__all__ = ["JsonRepairAgent"]
