
"""Asynchronous agent to review tailored resumes against job descriptions and professional profiles.
Outputs structured JSON feedback on match quality and issues."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from pydantic import ValidationError

from core.llm_client import AsyncLLMClient
from core.models import JDAnalysisResult, ProfessionalProfile, TailoredResume
from core.obs import Logger, NullLogger
from core.config import get_default_model
from agents.qa_shared import ResumeQAResult, QA_SYSTEM_PROMPT, QA_USER_TEMPLATE

class ResumeQAError(RuntimeError):
    """Base exception for Resume QA errors."""

class ResumeQAInvalid(ResumeQAError):
    """Indicates invalid or unparseable response from the QA model."""

@dataclass(slots=True)
class AsyncResumeQAAgent:
    """Asynchronous agent to review tailored resumes 
    against job descriptions and professional profiles.
    Outputs structured JSON feedback on match quality and issues.
    """

    llm: AsyncLLMClient
    model: str = field(default_factory=get_default_model)
    logger: Logger = field(default_factory=NullLogger)

    async def review(
        self,
        jd: JDAnalysisResult,
        profile: ProfessionalProfile,
        resume: TailoredResume,
    ) -> ResumeQAResult:
        """Reviews the tailored resume against the job description and professional profile."""
        msgs = [
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
        raw = await self.llm.chat(messages=msgs, model=self.model, temperature=0.0)
        self.logger.info("resume_qa.raw_response", raw=raw, raw_chars=len(raw))
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            s, e = raw.find("{"), raw.rfind("}")
            if s == -1 or e == -1 or e < s:
                raise ResumeQAInvalid("QA returned non-JSON response") from exc
            try:
                data = json.loads(raw[s : e + 1])
            except json.JSONDecodeError as exc2:
                raise ResumeQAInvalid("QA returned invalid JSON response") from exc2
        try:
            return ResumeQAResult.model_validate(data)
        except ValidationError as e:
            raise ResumeQAInvalid(f"QA validation failed: {e}") from e
