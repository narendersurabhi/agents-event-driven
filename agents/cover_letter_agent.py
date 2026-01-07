"""Agent for generating a tailored cover letter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from agents.common_prompts import build_composer_messages
from core.config import get_default_model
from core.json_utils import parse_json_object
from core.llm_client import LLMClient
from core.models import CoverLetter, JDAnalysisResult, ProfessionalProfile, TailoredResume


COVER_LETTER_SCHEMA_TEXT = """
from typing import Optional
from pydantic import BaseModel

class CoverLetter(BaseModel):
    full_name: str
    email: Optional[str]
    phone: Optional[str]
    company: Optional[str]
    role_title: Optional[str]
    body: str
"""

SYSTEM_PROMPT = f"""
You are a senior career copywriter specializing in concise, targeted cover letters
for technical roles. Given a job description analysis, a ProfessionalProfile,
and a final TailoredResume, you must generate a single, well-structured cover
letter in STRICT JSON matching the CoverLetter schema.

Requirements:
- Use the candidate's real full_name, email, and phone from the profile/resume.
- Address the letter to the company (and role_title) from the JD analysis when available.
- Focus on 3–5 short paragraphs:
  1) Opening: who you are and why you are a strong fit (avoid restating the exact role_title/company line verbatim, as this appears in the header).
  2) Alignment: how your experience and skills match the role/JD.
  3) Impact: 2–3 concrete achievements relevant to the JD.
  4) Closing: enthusiasm and call to action.
- Keep the tone professional, confident, and concise.
- Do NOT invent companies, roles, dates, or tools not present in the profile/resume.
- Do NOT fabricate certifications or degrees.
- If company or role_title are unknown, set them to null but still write a generic but strong cover letter.

Output STRICT JSON only, matching this schema:
{COVER_LETTER_SCHEMA_TEXT}
"""

USER_TEMPLATE = """
JD ANALYSIS:
{jd_json}

PROFESSIONAL PROFILE:
{profile_json}

TAILORED RESUME:
{resume_json}

Notes:
- Use profile.full_name and contact details as the candidate identity.
- Use jd.role_title and jd.company for the letter targeting when present, but avoid
  repeating the exact "I am applying for ROLE at COMPANY" phrasing that already
  appears in the heading of the document.
- The body field should contain the full cover letter text, including line breaks,
  starting directly with the first paragraph (no need to repeat the greeting or
  your full name inside body).

Return ONLY JSON for CoverLetter. No commentary.
"""


class CoverLetterError(RuntimeError):
    """Base error for cover letter generation."""


class CoverLetterInvalidResponse(CoverLetterError):
    """Raised when the cover letter JSON cannot be parsed/validated."""


@dataclass(slots=True)
class CoverLetterAgent:
    """Generate a CoverLetter from JD/profile/tailored resume."""

    llm: LLMClient
    model: str = field(default_factory=get_default_model)

    def build_messages(
        self,
        jd: JDAnalysisResult,
        profile: ProfessionalProfile,
        resume: TailoredResume,
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    jd_json=jd.model_dump_json(),
                    profile_json=profile.model_dump_json(),
                    resume_json=resume.model_dump_json(),
                ),
            },
        ]

    def parse_result(self, data: dict[str, Any]) -> CoverLetter:
        try:
            return CoverLetter.model_validate(data)
        except ValidationError as e:
            raise CoverLetterInvalidResponse(f"Validation failed: {e}") from e

    def generate(
        self,
        jd: JDAnalysisResult,
        profile: ProfessionalProfile,
        resume: TailoredResume,
    ) -> CoverLetter:
        msgs = self.build_messages(jd, profile, resume)
        raw = self.llm.chat(messages=msgs, model=self.model, temperature=0.2)
        data = parse_json_object(str(raw), CoverLetterInvalidResponse)
        return self.parse_result(data)
