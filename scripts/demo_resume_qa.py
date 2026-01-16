# scripts/demo_resume_qa.py
from agents.match_planner import MatchPlannerAgent
from agents.resume_composer import ResumeComposerAgent
from agents.resume_qa import ResumeQAAgent
from core.llm_client import OpenAILLMClient
from core.models import JDAnalysisResult, ProfessionalProfile


def main() -> None:
    """Run a basic QA pass using sample data (replace with your own)."""
    jd = JDAnalysisResult(
        role_title="Senior Data Engineer",
        company="ExampleCo",
        seniority_level="Senior",
        must_have_skills=["Python", "Spark", "AWS"],
        nice_to_have_skills=["Terraform"],
        notes_for_resume="Highlight scalable data pipelines and cloud infrastructure.",
    )
    profile = ProfessionalProfile(
        full_name="Sample Candidate",
        core_skills=["Python", "Spark", "AWS"],
        experience=[],
    )

    llm = OpenAILLMClient()
    plan = MatchPlannerAgent(llm).plan(jd, profile)
    tailored = ResumeComposerAgent(llm).compose(jd, profile, plan)
    qa = ResumeQAAgent(llm).review(jd, profile, tailored)
    print(qa.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
