# scripts/demo_resume_qa.py
from agents.resume_qa import ResumeQAAgent
from agents.match_planner import MatchPlannerAgent
from agents.resume_composer import ResumeComposerAgent
from core.llm_client import OpenAILLMClient
from core.models import JDAnalysisResult, ProfessionalProfile, ExperienceItem

# reuse your jd/profile from earlier
# ...
llm = OpenAILLMClient()
plan = MatchPlannerAgent(llm).plan(jd, profile)
tailored = ResumeComposerAgent(llm).compose(jd, profile, plan)
qa = ResumeQAAgent(llm).review(jd, profile, tailored)
print(qa.model_dump_json(indent=2))
