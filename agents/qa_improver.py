"""Synchronous QA-based Resume Improvement Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from pydantic import ValidationError

from agents.qa_improver_async import _SYSTEM as QA_IMPROVE_SYSTEM, _USER as QA_IMPROVE_USER
from agents.qa_shared import ResumeQAResult
from agents.resume_composer import COMPOSER_SCHEMA_TEXT
from core.config import get_default_model
from core.json_utils import parse_json_object
from core.llm_client import LLMClient
from core.models import JDAnalysisResult, ProfessionalProfile, TailoredResume
from core.obs import Logger, NullLogger


class QAImproveError(RuntimeError):
    """Base error for QA improvement failures."""


class QAImproveInvalid(QAImproveError):
    """Raised when the improved resume payload fails validation or parsing."""


@dataclass(slots=True)
class QAImproveAgent:
    """Synchronous agent that applies QA suggestions to improve a TailoredResume."""

    llm: LLMClient
    model: str = field(default_factory=get_default_model)
    logger: Logger = field(default_factory=NullLogger)

    def build_messages(
        self,
        jd: JDAnalysisResult,
        profile: ProfessionalProfile,
        resume: TailoredResume,
        qa: ResumeQAResult,
    ) -> list[dict[str, str]]:
        """Build chat messages for QA-based resume improvement."""
        return [
            {"role": "system", "content": QA_IMPROVE_SYSTEM},
            {
                "role": "user",
                "content": QA_IMPROVE_USER.format(
                    jd_json=jd.model_dump_json(),
                    profile_json=profile.model_dump_json(),
                    resume_json=resume.model_dump_json(),
                    suggestions_json=json.dumps(qa.suggestions, ensure_ascii=False, indent=2),
                ),
            },
        ]

    def parse_result(self, data: dict[str, Any]) -> TailoredResume:
        """Convert parsed JSON data into a TailoredResume."""
        try:
            return TailoredResume.model_validate(data)
        except ValidationError as e:
            raise QAImproveInvalid(f"Improved resume validation failed: {e}") from e

    def improve(
        self,
        jd: JDAnalysisResult,
        profile: ProfessionalProfile,
        resume: TailoredResume,
        qa: ResumeQAResult,
    ) -> TailoredResume:
        """Return an improved TailoredResume that addresses QA suggestions."""
        msgs = self.build_messages(jd, profile, resume, qa)
        raw = self.llm.chat(messages=msgs, model=self.model, temperature=0.1)
        self.logger.info("qa_improve_sync.raw_response", raw=raw, raw_chars=len(str(raw)))
        raw_str = str(raw)
        try:
            data = parse_json_object(raw_str, QAImproveInvalid)
        except QAImproveInvalid:
            # Fallback: try naive braces extraction then validate.
            start, end = raw_str.find("{"), raw_str.rfind("}")
            if start == -1 or end == -1:
                raise
            data = json.loads(raw_str[start : end + 1])
        return self.parse_result(data)


QA_IMPROVE_SCHEMA_TEXT = COMPOSER_SCHEMA_TEXT
