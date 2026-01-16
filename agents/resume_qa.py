"""Resume Quality Assurance Agent."""

# pylint: disable=R0801

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import ValidationError

from agents.qa_shared import QA_SYSTEM_PROMPT, QA_USER_TEMPLATE, ResumeQAResult
from core.config import get_default_model
from core.json_utils import parse_json_object
from core.llm_client import LLMClient
from core.models import JDAnalysisResult, ProfessionalProfile, TailoredResume


@dataclass(slots=True)
class ResumeQAAgent:
    """Synchronous QA over a tailored resume."""

    llm: LLMClient
    model: str = field(default_factory=get_default_model)

    def build_messages(
        self,
        jd: JDAnalysisResult,
        profile: ProfessionalProfile,
        resume: TailoredResume,
    ) -> list[dict[str, str]]:
        """Build chat messages for resume QA."""
        return [
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": QA_USER_TEMPLATE.format(
                    jd=jd.model_dump_json(),
                    profile=profile.model_dump_json(),
                    resume=resume.model_dump_json(),
                ),
            },
        ]

    def parse_result(self, data: dict[str, Any]) -> ResumeQAResult:
        """Convert parsed JSON data into a ResumeQAResult."""
        try:
            return ResumeQAResult.model_validate(data)
        except ValidationError as e:
            raise RuntimeError(f"QA validation failed: {e}") from e

    def review(
        self,
        jd: JDAnalysisResult,
        profile: ProfessionalProfile,
        resume: TailoredResume,
    ) -> ResumeQAResult:
        """Run QA and return a ResumeQAResult."""
        msgs = self.build_messages(jd, profile, resume)
        raw = cast(str, self.llm.chat(messages=msgs, model=self.model, temperature=0.0))
        data = parse_json_object(raw, RuntimeError)
        return self.parse_result(data)
