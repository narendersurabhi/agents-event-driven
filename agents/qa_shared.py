"""Shared QA models and prompts for resume QA agents."""

from __future__ import annotations

from typing import Dict, Literal, List
from pydantic import BaseModel, Field


class ResumeIssue(BaseModel):
    """QA issue detail."""

    severity: Literal["blocker", "major", "minor"]
    message: str
    location_hint: str | None = None  # e.g. "Summary", "Experience[0].bullets[2]"


class ResumeQAResult(BaseModel):
    """Structured QA output."""

    overall_match_score: float = Field(..., ge=0, le=100)
    must_have_coverage: Dict[str, bool]
    issues: List[ResumeIssue] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


QA_SCHEMA_TEXT = """
{
  "overall_match_score": number between 0-100,
  "must_have_coverage": {"skill_name": true|false, ...},
  "issues": [
    {"severity": "blocker"|"major"|"minor", "message": "string", "location_hint": "string | null"}
  ],
  "suggestions": ["string", ...]
}
"""


QA_SYSTEM_PROMPT = f"""
You are a strict reviewer. Output MUST be valid JSON for ResumeQAResult and nothing else.

ResumeQAResult schema (all fields required unless noted):
{QA_SCHEMA_TEXT}

Rules:
- must_have_coverage: key = JD must-have skill, value = boolean (true if covered; false if missing).
- issues: each item must include severity + message; include location_hint when known. Use issues for truthfulness gaps.
- suggestions: optional guidance strings.
- No additional top-level keys or nesting.
- No markdown or prose outside the JSON object.
"""

QA_USER_TEMPLATE = """
JD:
{jd}

PROFILE (truth source):
{profile}

TAILORED RESUME:
{resume}

Return only JSON matching ResumeQAResult.
"""
