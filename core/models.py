from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ==== Job description analysis output ====


class JDAnalysisResult(BaseModel):
    """Structured representation of a job description."""

    role_title: str = Field(..., description="Primary job title for this role.")
    company: str | None = Field(
        None,
        description="Company name if present or inferable from the job description.",
    )
    seniority_level: str | None = Field(
        None,
        description="Seniority, e.g. 'Senior', 'Staff', 'Lead', 'IC3'.",
    )
    must_have_skills: list[str] = Field(
        default_factory=list,
        description="Core skills required for the role.",
    )
    nice_to_have_skills: list[str] = Field(
        default_factory=list,
        description="Bonus/optional skills that strengthen a candidate.",
    )
    notes_for_resume: str = Field(
        "",
        description="Guidance on how to tailor a resume to this JD.",
    )


# ==== Candidate profile (canonical, JD-agnostic) ====


class ExperienceItem(BaseModel):
    title: str
    company: str
    start_date: str | None = None  # "Jan 2022"
    end_date: str | None = None  # "Present"
    location: str | None = None
    bullets: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)  # optional tagging


class ExperienceYearsClaim(BaseModel):
    """A verbatim years-of-experience claim from the resume text."""

    area: str  # e.g. "software engineering", "AI/ML", "data engineering"
    years_text: str  # e.g. "18+ years", "8+ years"
    evidence: str  # exact snippet from the resume containing the claim


class ProfessionalProfile(BaseModel):
    full_name: str
    headline: str | None = None
    location: str | None = None
    phone: str | None = None
    email: str | None = None
    linkedin_url: str | None = None
    github_url: str | None = None
    years_of_experience: float | None = None
    experience_years_claims: list[ExperienceYearsClaim] = Field(default_factory=list)
    core_skills: list[str] = Field(default_factory=list)
    domain_expertise: list[str] = Field(default_factory=list)
    tools_and_tech: list[str] = Field(default_factory=list)
    experience: list[ExperienceItem] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    education_items: list[EducationItem] = Field(default_factory=list)
    certifications: list[CertificationItem] = Field(default_factory=list)


# ==== Planning outputs ====


class ExperiencePlan(BaseModel):
    profile_experience_index: int
    include: bool
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    target_bullet_count: int = Field(..., ge=0)
    focus_skills: list[str] = Field(default_factory=list)


class SkillsPlan(BaseModel):
    must_have_covered: list[str] = Field(default_factory=list)
    must_have_missing: list[str] = Field(default_factory=list)
    nice_to_have_covered: list[str] = Field(default_factory=list)
    extra_profile_skills: list[str] = Field(default_factory=list)


class ResumePlan(BaseModel):
    target_title: str
    target_company: str | None = None
    sections_order: list[str] = Field(
        default_factory=lambda: ["Summary", "Skills", "Experience", "Education"]
    )
    length_hint: Literal["one_page", "two_pages_ok"] = "one_page"
    experiences_plan: list[ExperiencePlan] = Field(default_factory=list)
    skills_plan: SkillsPlan = Field(default_factory=SkillsPlan)


# ==== Tailored resume output ====


class TailoredBullet(BaseModel):
    text: str
    source_experience_index: int | None = Field(
        None, description="Index into ProfessionalProfile.experience for traceability."
    )


class TailoredExperienceItem(BaseModel):
    title: str
    company: str
    start_date: str | None = None
    end_date: str | None = None
    location: str | None = None
    bullets: list[TailoredBullet] = Field(default_factory=list)


class SkillCategory(BaseModel):
    """Grouped skills, e.g. 'ML / AI & LLMs'."""

    name: str
    items: list[str] = Field(default_factory=list)


class EducationItem(BaseModel):
    institution: str
    degree: str
    start_date: str | None = None
    end_date: str | None = None
    location: str | None = None


class CertificationItem(BaseModel):
    name: str
    issuer: str | None = None
    year: str | None = None


class TailoredResume(BaseModel):
    full_name: str
    headline: str | None = None
    location: str | None = None
    phone: str | None = None
    email: str | None = None
    linkedin_url: str | None = None
    github_url: str | None = None

    summary: str
    skills: list[SkillCategory] = Field(default_factory=list)
    experience: list[TailoredExperienceItem] = Field(default_factory=list)
    education: list[EducationItem] = Field(default_factory=list)
    certifications: list[CertificationItem] = Field(default_factory=list)

    # Optional flattened text/Markdown version for download/preview
    resume_text: str | None = None


# ==== Cover letter output ====


class CoverLetter(BaseModel):
    """Structured cover letter for a specific role/company."""

    full_name: str
    email: str | None = None
    phone: str | None = None
    company: str | None = None
    role_title: str | None = None
    body: str
