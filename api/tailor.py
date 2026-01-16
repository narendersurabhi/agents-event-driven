import os

from fastapi import APIRouter
from pydantic import BaseModel

from agents.match_planner_async import AsyncMatchPlannerAgent
from agents.resume_composer_async import AsyncResumeComposerAgent
from agents.resume_qa_async import AsyncResumeQAAgent, ResumeQAResult
from core.llm_factory import get_async_llm_client
from core.models import JDAnalysisResult, ProfessionalProfile, TailoredResume
from core.obs import JsonRepoLogger

router = APIRouter()

APP_ENV: str = os.getenv("APP_ENV", "dev")
SERVICE_NAME: str = os.getenv("SERVICE_NAME", "tailor-api")

# ----- startup -----
logger = JsonRepoLogger(service=SERVICE_NAME, env=APP_ENV)

# Single shared async LLM client with DI-friendly logger
_llm = get_async_llm_client(logger=logger)

_planner = AsyncMatchPlannerAgent(llm=_llm)
_composer = AsyncResumeComposerAgent(llm=_llm)
_qa = AsyncResumeQAAgent(llm=_llm, logger=logger)


class TailorRequest(BaseModel):
    jd: JDAnalysisResult
    profile: ProfessionalProfile
    run_qa: bool = True  # wire QA later if you want


class TailorResponse(BaseModel):
    tailored: TailoredResume
    qa: ResumeQAResult | None = None  # add when you convert QA to async


@router.post("/tailor-resume", response_model=TailorResponse)
async def tailor_resume(payload: TailorRequest) -> TailorResponse:
    plan = await _planner.plan(payload.jd, payload.profile)
    tailored = await _composer.compose(payload.jd, payload.profile, plan)
    qa = None
    if payload.run_qa:
        qa = await _qa.review(payload.jd, payload.profile, tailored)
    return TailorResponse(tailored=tailored, qa=qa)
