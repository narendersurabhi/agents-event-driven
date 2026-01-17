"""Shared enums for GovGuard contracts."""

from __future__ import annotations

from enum import Enum


class ArtifactType(str, Enum):
    """Artifact types tracked by governance."""

    MODEL = "MODEL"
    PROMPT = "PROMPT"
    RETRIEVAL_CONFIG = "RETRIEVAL_CONFIG"
    POLICY = "POLICY"


class Environment(str, Enum):
    """Deployment environments."""

    DEV = "dev"
    STAGE = "stage"
    PROD = "prod"


class GateDecisionType(str, Enum):
    """Gatekeeper decisions."""

    APPROVE = "APPROVE"
    APPROVE_WITH_WARNINGS = "APPROVE_WITH_WARNINGS"
    BLOCK = "BLOCK"
