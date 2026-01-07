"""Shared prompt builders for agents."""

from __future__ import annotations

from core.models import JDAnalysisResult, ProfessionalProfile, ResumePlan


def build_jd_analysis_messages(
    system_prompt: str, user_template: str, job_description: str
) -> list[dict[str, str]]:
    """Construct messages for JD analysis."""
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_template.format(job_description=job_description.strip()),
        },
    ]


def build_match_planner_messages(
    system_prompt: str,
    user_template: str,
    jd: JDAnalysisResult,
    profile: ProfessionalProfile,
) -> list[dict[str, str]]:
    """Construct messages for match planning."""
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_template.format(
                jd_json=jd.model_dump_json(),
                profile_json=profile.model_dump_json(),
            ),
        },
    ]


def build_composer_messages(
    system_prompt: str,
    user_template: str,
    jd: JDAnalysisResult,
    profile: ProfessionalProfile,
    plan: ResumePlan,
) -> list[dict[str, str]]:
    """Construct messages for resume composition."""
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_template.format(
                jd_json=jd.model_dump_json(),
                profile_json=profile.model_dump_json(),
                plan_json=plan.model_dump_json(),
            ),
        },
    ]
