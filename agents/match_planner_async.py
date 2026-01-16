"""Async match planning agent."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

from pydantic import ValidationError

from core.config import get_default_model
from core.llm_client import AsyncLLMClient
from core.models import JDAnalysisResult, ProfessionalProfile, ResumePlan

from .match_planner import (
    MatchPlanError,
    MatchPlanInvalidResponse,
    _MatchPlannerShared,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AsyncMatchPlannerAgent(_MatchPlannerShared):
    """Asynchronously generate a ResumePlan for a JD/profile."""

    llm: AsyncLLMClient
    model: str = field(default_factory=get_default_model)

    async def plan(self, jd: JDAnalysisResult, profile: ProfessionalProfile) -> ResumePlan:
        """Generate a ResumePlan asynchronously."""
        if not profile.experience:
            raise MatchPlanError("Profile has no experience to plan")

        msgs = self._messages(jd, profile)
        logger.info("AsyncMatchPlannerAgent: planning for %s", jd.role_title)
        raw = await self.llm.chat(messages=msgs, model=self.model, temperature=0.1)
        data = self._parse_json(raw)
        try:
            plan: ResumePlan = ResumePlan.model_validate(data)
        except ValidationError as e:
            raise MatchPlanInvalidResponse(f"Validation failed: {e}") from e
        return plan
