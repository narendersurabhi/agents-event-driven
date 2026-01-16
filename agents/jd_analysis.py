""" Job Description Analysis Agent """

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, cast

from pydantic import ValidationError

from agents.common_prompts import build_jd_analysis_messages
from core.config import get_default_model
from core.json_utils import parse_json_object
from core.llm_client import LLMClient
from core.models import JDAnalysisResult

logger = logging.getLogger(__name__)


JD_SCHEMA_TEXT = """
You are a precise Job Description Analysis Agent.

You receive one job description as plain text.
You MUST respond with STRICT JSON ONLY that matches this Python model:

class JDAnalysisResult(BaseModel):
    role_title: str
    company: Optional[str]
    seniority_level: Optional[str]
    must_have_skills: List[str]
    nice_to_have_skills: List[str]
    notes_for_resume: str

"""

AGENT_SYSTEM_PROMPT = f"""
You are a precise Job Description Analysis Agent.

You receive one job description as plain text.
You MUST respond with STRICT JSON ONLY that matches this Python model:

{JD_SCHEMA_TEXT}

Guidelines:
- Infer role_title, company, and seniority_level from the text when possible.
- must_have_skills = core technical and domain skills explicitly required or clearly implied.
- nice_to_have_skills = secondary or "bonus" skills.
- notes_for_resume = concise bullet-style text (as one string) with suggestions on how a candidate should tailor their resume.

VERY IMPORTANT:
- Output MUST be valid JSON.
- Do NOT include comments or explanations.
- Do NOT wrap in Markdown.
"""

USER_PROMPT_TEMPLATE = """
Here is the job description:

---
{job_description}
---

Return ONLY JSON for JDAnalysisResult.
"""


class JDAnalysisError(RuntimeError):
    """Base error for JD analysis agent."""


class JDAnalysisInvalidResponse(JDAnalysisError):
    """Raised when the LLM output cannot be parsed or validated."""


@dataclass(slots=True)
class JDAnalysisAgent:
    """Single-responsibility agent for analyzing job descriptions (sync).

    Responsibilities:
    - Accept a JD string.
    - Call an LLM with a strongly worded system prompt.
    - Parse raw JSON and validate into JDAnalysisResult.
    - Raise well-typed errors if anything goes wrong.
    """

    llm: LLMClient
    model: str = field(default_factory=get_default_model)

    def build_messages(self, job_description: str) -> list[dict[str, str]]:
        """Build chat messages for JD analysis."""
        return build_jd_analysis_messages(
            AGENT_SYSTEM_PROMPT,
            USER_PROMPT_TEMPLATE,
            job_description,
        )

    def _parse_json(self, raw: str) -> dict[str, Any]:
        """Best-effort JSON extraction + parsing with logging."""
        try:
            return parse_json_object(raw, JDAnalysisInvalidResponse)
        except JDAnalysisInvalidResponse:
            logger.error("JDAnalysisAgent: failed to parse JSON")
            raise

    def parse_result(self, data: dict[str, Any]) -> JDAnalysisResult:
        """Convert parsed JSON data into a JDAnalysisResult."""
        try:
            result: JDAnalysisResult = JDAnalysisResult.model_validate(data)
        except ValidationError as e:
            logger.exception("JDAnalysisAgent: validation failed")
            raise JDAnalysisInvalidResponse(f"Validation failed: {e}") from e

        logger.info(
            "JDAnalysisAgent: success role_title=%s company=%s",
            result.role_title,
            result.company,
        )
        return result

    def analyze(self, job_description: str) -> JDAnalysisResult:
        """Analyze a job description into a structured JDAnalysisResult."""
        jd = job_description.strip()
        if not jd:
            raise JDAnalysisError("job_description must be a non-empty string")

        messages = self.build_messages(jd)

        logger.info("JDAnalysisAgent: analyzing JD (%d chars)", len(jd))

        raw_obj = self.llm.chat(messages=messages, model=self.model, temperature=0.1)
        raw: str = cast(str, raw_obj)
        logger.debug("JDAnalysisAgent: raw LLM output: %s", raw[:500])

        data = self._parse_json(raw)

        return self.parse_result(data)
