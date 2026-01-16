"""Async resume composer."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

from pydantic import ValidationError

from core.config import get_default_model
from core.llm_client import AsyncLLMClient
from core.models import JDAnalysisResult, ProfessionalProfile, ResumePlan, TailoredResume

from .resume_composer import ComposeInvalidResponse, _ComposerShared

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AsyncResumeComposerAgent(_ComposerShared):
    """Compose a TailoredResume asynchronously."""

    llm: AsyncLLMClient
    model: str = field(default_factory=get_default_model)

    async def compose(
        self,
        jd: JDAnalysisResult,
        profile: ProfessionalProfile,
        plan: ResumePlan,
    ) -> TailoredResume:
        """Compose a TailoredResume asynchronously."""
        msgs = self._messages(jd, profile, plan)
        logger.info("AsyncResumeComposerAgent: composing for %s @ %s", jd.role_title, jd.company)
        raw = await self.llm.chat(messages=msgs, model=self.model, temperature=0.1)
        data = self._parse_json(raw)
        try:
            return TailoredResume.model_validate(data)
        except ValidationError as e:
            raise ComposeInvalidResponse(f"Validation failed: {e}") from e
