
"""Extract ProfessionalProfile from raw resume text."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import ValidationError

from core.config import get_default_model
from core.json_utils import parse_json_object
from core.llm_client import LLMClient
from core.models import ProfessionalProfile

logger = logging.getLogger(__name__)

PROFILE_SCHEMA_TEXT = """
class ExperienceItem(BaseModel):
    title: str
    company: str
    start_date: Optional[str]
    end_date: Optional[str]
    location: Optional[str]
    bullets: List[str]
    skills: List[str]

class EducationItem(BaseModel):
    institution: str
    degree: str
    start_date: Optional[str]
    end_date: Optional[str]
    location: Optional[str]

class CertificationItem(BaseModel):
    name: str
    issuer: Optional[str]
    year: Optional[str]

class ExperienceYearsClaim(BaseModel):
    area: str
    years_text: str
    evidence: str

class ProfessionalProfile(BaseModel):
    full_name: str
    headline: Optional[str]
    location: Optional[str]
    phone: Optional[str]
    email: Optional[str]
    linkedin_url: Optional[str]
    github_url: Optional[str]
    years_of_experience: Optional[float]
    experience_years_claims: List[ExperienceYearsClaim]
    core_skills: List[str]
    domain_expertise: List[str]
    tools_and_tech: List[str]
    experience: List[ExperienceItem]
    education: List[str]
    education_items: List[EducationItem]
    certifications: List[CertificationItem]
"""

_SYSTEM = f"""
You are a Resume → ProfessionalProfile extractor.
Given a resume as raw text, output STRICT JSON matching this schema:

{PROFILE_SCHEMA_TEXT}

Rules:
- Do NOT invent facts. If unknown, use null or empty list.
- Only mention years if they appear verbatim in the resume text. Do NOT compute or
  infer years (e.g., total years_of_experience) from dates or timelines. If the
  resume does not explicitly state a total years-of-experience number, set
  years_of_experience to null.
- experience_years_claims: extract any explicit "X+ years" style claims that appear
  verbatim in the resume (e.g. "18+ years of software engineering experience",
  "8+ years building ML/AI systems"). For each claim:
  - years_text must be exactly the "X+ years" fragment as written.
  - evidence must be an exact snippet from the resume containing the claim.
  - area should be a short label for what the claim applies to (e.g. "software engineering", "AI/ML").
- Bullets should be concise, factual, and come from the resume text.
- skills lists should reflect tools/tech explicitly seen in the resume.
- For education_items, extract structured entries (institution, degree, dates, location)
  when possible from the resume's education section; otherwise leave the list empty
  and keep raw strings in education.
- For certifications, only extract real certifications explicitly present in the
  resume (name, issuer, year when available); do NOT invent certifications.
- For phone/email/linkedin_url/github_url, copy them exactly from the resume header
  if present; otherwise use null and do NOT fabricate or guess values.
- Output JSON only, no markdown or commentary.
"""

_USER = "RESUME TEXT:\n---\n{resume_text}\n---\nReturn only ProfessionalProfile JSON."

class ProfileExtractError(RuntimeError):
    """Base error for profile extraction."""


class ProfileExtractInvalid(ProfileExtractError):
    """Raised when extractor output cannot be parsed or validated."""

@dataclass(slots=True)
class ProfileFromResumeAgent:
    """Extract a ProfessionalProfile from resume text."""

    llm: LLMClient
    model: str = field(default_factory=get_default_model)

    def build_messages(self, resume_text: str) -> list[dict[str, str]]:
        """Build chat messages for extracting a profile from resume_text."""
        return [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _USER.format(resume_text=resume_text.strip())},
        ]

    def parse_result(self, data: dict[str, Any]) -> ProfessionalProfile:
        """Convert parsed JSON data into a ProfessionalProfile."""
        try:
            return ProfessionalProfile.model_validate(data)
        except ValidationError as e:
            raise ProfileExtractInvalid(f"Validation failed: {e}") from e

    def extract(self, resume_text: str) -> ProfessionalProfile:
        """Return a ProfessionalProfile parsed from resume_text.

        This method performs a direct LLM call plus JSON parse and validation.
        Centralized orchestration can instead use build_messages + parse_result
        together with the shared LLM step worker and repair logic.
        """
        msgs = self.build_messages(resume_text)
        raw = cast(str, self.llm.chat(messages=msgs, model=self.model, temperature=0.0))
        data = parse_json_object(raw, ProfileExtractInvalid)
        return self.parse_result(data)
