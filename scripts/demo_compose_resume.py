from agents.match_planner import MatchPlannerAgent
from agents.resume_composer import ResumeComposerAgent
from core.llm_client import OpenAILLMClient
from core.models import ExperienceItem, JDAnalysisResult, ProfessionalProfile

# Sample JD + profile (reuse from your match plan demo)
jd = JDAnalysisResult(
    role_title="Senior ML Engineer",
    company="Netflix",
    seniority_level="Senior",
    must_have_skills=["Python", "PySpark", "AWS", "MLOps", "Experimentation"],
    nice_to_have_skills=["Kubernetes", "Feature Store", "LLM/GenAI"],
    notes_for_resume="Emphasize large-scale recsys/personalization.",
)

profile = ProfessionalProfile(
    full_name="Jane Doe",
    headline="Senior ML Engineer | Recsys | MLOps",
    core_skills=["Python", "Spark", "AWS", "MLOps", "Feature Store", "Metrics"],
    domain_expertise=["Recommenders", "Healthcare", "Fraud Analytics"],
    tools_and_tech=["SageMaker", "Airflow", "Docker", "K8s"],
    experience=[
        ExperienceItem(
            title="Senior ML Engineer",
            company="StreamCo",
            start_date="Jan 2022",
            end_date="Present",
            location="Remote",
            bullets=[
                "Built ranking models for homepage personalization",
                "Deployed pipelines on AWS with SageMaker and Airflow",
                "Ran A/B tests to evaluate model uplift on CTR",
            ],
            skills=["Python", "Spark", "AWS", "SageMaker", "A/B testing", "Recsys"],
        ),
        ExperienceItem(
            title="ML Engineer",
            company="HealthCo",
            start_date="2019",
            end_date="2021",
            bullets=[
                "Fraud detection on claims using gradient boosting",
                "Spark pipelines, model monitoring, weekly retrains",
            ],
            skills=["Python", "Spark", "AWS", "XGBoost", "Monitoring"],
        ),
    ],
    education=["MS Computer Science"],
)

llm = OpenAILLMClient()
planner = MatchPlannerAgent(llm=llm)
plan = planner.plan(jd, profile)

composer = ResumeComposerAgent(llm=llm)
tailored = composer.compose(jd, profile, plan)

print("=== Tailored JSON ===")
print(tailored.model_dump_json(indent=2))
print("\n=== Flattened resume_text ===")
print(tailored.resume_text)
