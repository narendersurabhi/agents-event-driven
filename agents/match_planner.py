"""Match planning agent."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

from pydantic import ValidationError

from agents.common_prompts import build_match_planner_messages
from core.config import get_default_model
from core.json_utils import parse_json_object
from core.llm_client import LLMClient
from core.models import JDAnalysisResult, ProfessionalProfile, ResumePlan

logger = logging.getLogger(__name__)

MATCH_PLAN_SCHEMA_TEXT = """
class ResumePlan(BaseModel):
    target_title: str
    target_company: Optional[str]
    sections_order: list[str]
    length_hint: Literal["one_page","two_pages_ok"]
    experiences_plan: list[ExperiencePlan]
    skills_plan: SkillsPlan

class ExperiencePlan(BaseModel):
    profile_experience_index: int
    include: bool
    relevance_score: float
    target_bullet_count: int
    focus_skills: list[str]

class SkillsPlan(BaseModel):
    must_have_covered: list[str]
    must_have_missing: list[str]
    nice_to_have_covered: list[str]
    extra_profile_skills: list[str]
"""

SYSTEM_PROMPT = f"""
You are the Match Planner. Given a JD analysis and a candidate's professional profile,
produce a concrete plan for how the resume SHOULD be tailored. You DO NOT write text.
You ONLY output a JSON plan that matches the given ResumePlan schema.

Rules:
- Include only experiences from the profile (by index).
- Choose which experiences to include and how many bullets each gets.
- Focus on covering JD must-have skills with evidence from the profile.
- Set length_hint: "one_page" if 0–7 years of exp or small scope; otherwise "two_pages_ok".
- Do NOT invent skills or companies. Planning only.
- Scores are 0.0–1.0 where 1.0 means highly relevant to the JD.

Output STRICT JSON, no markdown, matching:

{MATCH_PLAN_SCHEMA_TEXT}
"""

USER_TEMPLATE = """
JD ANALYSIS:
{jd_json}

CANDIDATE PROFILE:
{profile_json}

Return ONLY JSON for ResumePlan. No commentary.
"""


class MatchPlanError(RuntimeError):
    """Base error for match planning."""


class MatchPlanInvalidResponse(MatchPlanError):
    """Raised when the LLM output cannot be parsed or validated."""


@dataclass(slots=True)
class _MatchPlannerShared:  # pylint: disable=too-few-public-methods
    """Reusable helpers for match planner agents."""

    def _messages(
        self,
        jd: JDAnalysisResult,
        profile: ProfessionalProfile,
    ) -> list[dict[str, str]]:
        return build_match_planner_messages(SYSTEM_PROMPT, USER_TEMPLATE, jd, profile)

    def _parse_json(self, raw: str) -> dict[str, Any]:
        return parse_json_object(raw, MatchPlanInvalidResponse)


@dataclass(slots=True)
class MatchPlannerAgent(_MatchPlannerShared):
    """Plan which experiences/skills to emphasize for a JD."""

    llm: LLMClient
    model: str = field(default_factory=get_default_model)

    def build_messages(
        self, jd: JDAnalysisResult, profile: ProfessionalProfile
    ) -> list[dict[str, str]]:
        """Build chat messages for match planning."""
        return self._messages(jd, profile)

    def parse_result(self, data: dict[str, Any]) -> ResumePlan:
        """Convert parsed JSON data into a ResumePlan."""
        try:
            return ResumePlan.model_validate(data)
        except ValidationError as e:
            raise MatchPlanInvalidResponse(f"Validation failed: {e}") from e

    def plan(self, jd: JDAnalysisResult, profile: ProfessionalProfile) -> ResumePlan:
        """Generate a ResumePlan for the given JD/profile."""
        if not profile.experience:
            raise MatchPlanError("Profile has no experience to plan")

        msgs = self._messages(jd, profile)
        logger.info("MatchPlannerAgent: planning for %s", jd.role_title)
        raw: str = self.llm.chat(messages=msgs, model=self.model, temperature=0.1)
        data = self._parse_json(raw)
        return self.parse_result(data)
