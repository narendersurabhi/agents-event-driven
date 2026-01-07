"""Synchronous resume composer."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from agents.common_prompts import build_composer_messages
from core.config import get_default_model
from core.json_utils import parse_json_object
from core.llm_client import LLMClient
from core.models import JDAnalysisResult, ProfessionalProfile, ResumePlan, TailoredResume

logger = logging.getLogger(__name__)

COMPOSER_SCHEMA_TEXT = """
from typing import List, Optional
from pydantic import BaseModel

class TailoredResume(BaseModel):
    # Header
    full_name: str
    headline: Optional[str]
    location: Optional[str]
    phone: Optional[str]
    email: Optional[str]
    linkedin_url: Optional[str]
    github_url: Optional[str]

    # Core sections
    summary: str
    skills: List[SkillCategory]
    experience: List[TailoredExperienceItem]
    education: List[EducationItem]
    certifications: List[CertificationItem]

    # Optional fully rendered text/markdown version
    resume_text: Optional[str]

class SkillCategory(BaseModel):
    name: str               # e.g. "ML / AI & LLMs"
    items: List[str]        # e.g. ["Supervised & unsupervised ML", "LLMs: Bedrock, LangGraph, RAG"]

class TailoredExperienceItem(BaseModel):
    title: str              # e.g. "Senior Machine Learning Engineer & AI Solutions Architect"
    company: str            # e.g. "Acentra Health"
    start_date: Optional[str]   # e.g. "Dec 2016"
    end_date: Optional[str]     # e.g. "Present"
    location: Optional[str]     # e.g. "Okemos, MI"
    bullets: List[TailoredBullet]

class TailoredBullet(BaseModel):
    text: str
    source_experience_index: Optional[int]

class EducationItem(BaseModel):
    institution: str        # e.g. "Nizam College"
    degree: str             # e.g. "B.Sc., Mathematics, Physics, Electronics"
    start_date: Optional[str]   # e.g. "Sep 2003"
    end_date: Optional[str]     # e.g. "May 2007"
    location: Optional[str]     # e.g. "Hyderabad, India"

class CertificationItem(BaseModel):
    name: str               # e.g. "AWS Certified AI Practitioner (AIF-C01)"
    issuer: Optional[str]   # e.g. "Amazon Web Services"
    year: Optional[str]     # e.g. "2025"
"""

SYSTEM_PROMPT = f"""
You are the Resume Composer. You generate a tailored, ATS-friendly resume TEXT and JSON
strictly from the candidate's ProfessionalProfile and a ResumePlan derived from the JD.
You MUST NOT invent facts, companies, dates, or tools not present in the profile.
You MAY rephrase and reorder content to better match the JD, but stay truthful.

Constraints:
- Follow ResumePlan.experiences_plan for which experiences to include, order, and target_bullet_count.
- Emphasize JD must-have skills (when they are genuinely supported by the profile).
- Prefer one-column, simple sections: Summary, Skills, Experience, Education, Certifications.
- Headline: write a concise, JD-targeted title that reflects the target role and seniority
  (e.g. based on jd.role_title / ResumePlan.target_title), do not simply copy the original resume headline.
- Summary: 2-4 lines, align to JD target_title/seniority.
  - Do not invent years. Only use numeric year claims if they appear in the
    ProfessionalProfile JSON (years_of_experience or experience_years_claims evidence).
  - If the profile includes both an overall years claim and a domain-specific claim
    relevant to the target role, you may phrase as:
      "X+ years of overall experience, including Y+ years in <target-relevant area>."
- Bullets: concrete results, metrics/scale where present in profile; no fluff.
- Skills section: populate TailoredResume.skills as a small set of SkillCategory groups
  (e.g. "ML / AI & LLMs", "Data Platforms & MLOps", "Languages & Frameworks"). Each
  SkillCategory.items should be short, concrete skill descriptors. Include JD must-haves
  and include supporting evidence in the experience bullets.
- Education: map profile.education into structured EducationItem entries where possible.
- Certifications: only include real, evidenced certifications; leave empty if unknown.
- No tables/columns/graphics; plain text only.

Output STRICT JSON only, matching TailoredResume:
{COMPOSER_SCHEMA_TEXT}
"""

USER_TEMPLATE = """
JD ANALYSIS:
{jd_json}

PROFESSIONAL PROFILE:
{profile_json}

RESUME PLAN:
{plan_json}

Notes:
- Use profile.full_name and keep factual details consistent with the profile.
- You may and should rewrite the headline to better match the JD's target role and seniority
  (while remaining truthful to the candidate's experience).
- Do NOT fabricate emails/phones/links; leave them null if unknown.
- For each included experience, reuse title/company/dates/location from profile.
- Generate resume_text as a clean, ATS-friendly plain text or Markdown version of the resume.

Return ONLY JSON for TailoredResume. No commentary.
"""

class ComposeError(RuntimeError):
    """Base error for composition."""


class ComposeInvalidResponse(ComposeError):
    """Raised when the composer output cannot be parsed/validated."""

@dataclass(slots=True)
class _ComposerShared:  # pylint: disable=too-few-public-methods
    """Shared helpers for resume composer agents."""

    def _messages(
        self,
        jd: JDAnalysisResult,
        profile: ProfessionalProfile,
        plan: ResumePlan,
    ) -> list[dict[str, str]]:
        return build_composer_messages(SYSTEM_PROMPT, USER_TEMPLATE, jd, profile, plan)

    def _parse_json(self, raw: str) -> dict[str, Any]:
        return parse_json_object(raw, ComposeInvalidResponse)


@dataclass(slots=True)
class ResumeComposerAgent(_ComposerShared):
    """Compose a TailoredResume from JD/profile/plan."""

    llm: LLMClient
    model: str = field(default_factory=get_default_model)

    def build_messages(
        self,
        jd: JDAnalysisResult,
        profile: ProfessionalProfile,
        plan: ResumePlan,
    ) -> list[dict[str, str]]:
        """Build chat messages for resume composition."""
        return self._messages(jd, profile, plan)

    def parse_result(self, data: dict[str, Any]) -> TailoredResume:
        """Convert parsed JSON data into a TailoredResume."""
        try:
            return TailoredResume.model_validate(data)
        except ValidationError as e:
            raise ComposeInvalidResponse(f"Validation failed: {e}") from e

    def compose(
        self,
        jd: JDAnalysisResult,
        profile: ProfessionalProfile,
        plan: ResumePlan,
    ) -> TailoredResume:
        """Compose a TailoredResume synchronously."""
        msgs = self._messages(jd, profile, plan)
        logger.info("ResumeComposerAgent: composing for %s @ %s", jd.role_title, jd.company)
        raw: str = self.llm.chat(messages=msgs, model=self.model, temperature=0.1)
        data = self._parse_json(raw)
        return self.parse_result(data)
