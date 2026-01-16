import pytest

from agents.resume_qa_async import AsyncResumeQAAgent
from core.models import (
    EducationItem,
    ExperienceItem,
    JDAnalysisResult,
    ProfessionalProfile,
    TailoredBullet,
    TailoredExperienceItem,
    TailoredResume,
)

# --------- Fakes ---------


class FakeAsyncLLM_QA_Good:
    """Returns a valid QA JSON payload."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
    ) -> str:
        # Minimal-but-valid QA output
        return """
        {
          "overall_match_score": 86.0,
          "must_have_coverage": {
            "Python": true,
            "MLOps": true,
            "LLMs": false
          },
          "issues": [
            {
              "severity": "major",
              "message": "LLMs listed in skills but not supported by any experience bullet.",
              "location_hint": "Skills"
            }
          ],
          "suggestions": [
            "Add a bullet under most recent role describing concrete LLM usage or remove from skills."
          ]
        }
        """


class FakeAsyncLLM_QA_BadJSON:
    """Returns non-JSON to test error handling."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
    ) -> str:
        return "not json at all"


# --------- Fixtures ---------


@pytest.fixture
def sample_jd() -> JDAnalysisResult:
    return JDAnalysisResult(
        role_title="Senior ML Engineer - AI Platform",
        company="Coinbase",
        seniority_level="Senior",
        must_have_skills=["Python", "MLOps", "LLMs"],
        nice_to_have_skills=["Kafka", "Databricks"],
        notes_for_resume="Emphasize platform, production ML services, and security.",
    )


@pytest.fixture
def sample_profile() -> ProfessionalProfile:
    return ProfessionalProfile(
        full_name="Narender Surabhi",
        headline="Senior AI/ML Engineer | Spark/PySpark | MLOps on AWS",
        location="Okemos, MI",
        years_of_experience=15,
        core_skills=["Python", "Spark", "PySpark", "AWS", "SageMaker", "MLflow", "MLOps", "LLMs"],
        domain_expertise=["Healthcare claims", "FWA detection"],
        tools_and_tech=["SageMaker", "Glue", "Redshift", "Kafka"],
        experience=[
            ExperienceItem(
                title="AI/ML Specialist",
                company="Acentra Health",
                start_date="Dec 2016",
                end_date="Present",
                location="Okemos, MI",
                bullets=[
                    "Built distributed ML pipelines with Spark and SageMaker processing 100M+ annual claims.",
                    "Implemented monitoring with Clarify; drift/bias dashboards.",
                ],
                skills=["Python", "Spark", "AWS", "SageMaker", "MLflow", "MLOps"],
            )
        ],
        education=["B.Sc. (Math/Physics/Electronics) - Nizam College"],
    )


@pytest.fixture
def sample_tailored(sample_profile: ProfessionalProfile) -> TailoredResume:
    return TailoredResume(
        full_name=sample_profile.full_name,
        headline=sample_profile.headline,
        location=sample_profile.location,
        phone=None,
        email=None,
        linkedin_url=None,
        github_url=None,
        summary="Senior ML Engineer specializing in platform-scale ML pipelines and MLOps on AWS.",
        skills=[],
        experience=[
            TailoredExperienceItem(
                title=sample_profile.experience[0].title,
                company=sample_profile.experience[0].company,
                start_date=sample_profile.experience[0].start_date,
                end_date=sample_profile.experience[0].end_date,
                location=sample_profile.experience[0].location,
                bullets=[
                    TailoredBullet(
                        text="Built Spark + SageMaker pipelines for 100M+ annual claims in production.",
                        source_experience_index=0,
                    ),
                    TailoredBullet(
                        text="Added drift/bias dashboards with Clarify; improved governance and reliability.",
                        source_experience_index=0,
                    ),
                ],
            )
        ],
        education=[
            # Map old simple education string into the new structured form.
            EducationItem(
                institution="Nizam College",
                degree="B.Sc. (Math/Physics/Electronics)",
                start_date=None,
                end_date=None,
                location=None,
            )
        ],
        certifications=[],
        resume_text="(flattened text omitted for test)",
    )


# --------- Tests ---------


@pytest.mark.asyncio
async def test_resume_qa_happy_path(sample_jd, sample_profile, sample_tailored):
    agent = AsyncResumeQAAgent(llm=FakeAsyncLLM_QA_Good(), model="test-model")
    result = await agent.review(sample_jd, sample_profile, sample_tailored)

    assert 0 <= result.overall_match_score <= 100
    assert "Python" in result.must_have_coverage
    assert isinstance(result.issues, list)
    assert any(i.severity in {"blocker", "major", "minor"} for i in result.issues)
    assert isinstance(result.suggestions, list)


@pytest.mark.asyncio
async def test_resume_qa_bad_json_raises(sample_jd, sample_profile, sample_tailored):
    from agents.resume_qa_async import ResumeQAInvalid

    agent = AsyncResumeQAAgent(llm=FakeAsyncLLM_QA_BadJSON(), model="test-model")
    with pytest.raises(ResumeQAInvalid):
        await agent.review(sample_jd, sample_profile, sample_tailored)
