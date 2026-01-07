"""Demo script: full event-driven resume pipeline.

Pipeline:
  jd.txt + resume.txt
    → jd.requested      → JDWorker + LLMStepWorker → jd.completed
    → profile.requested → ProfileWorker + LLMStepWorker → profile.completed
    → match.requested   → MatchWorker + LLMStepWorker → match.completed
    → compose.requested → ResumeComposerWorker + LLMStepWorker → compose.completed

All communication is via InMemoryEventBus events; agents never call each
other directly, and JSON repair is centralized in LLMStepWorker.

Usage:
  python -m scripts.run_full_event_pipeline jd.txt resume.txt
"""

from __future__ import annotations

import json
import logging
import threading
from argparse import ArgumentParser, FileType

from agents.jd_analysis import JDAnalysisAgent
from agents.jd_worker import JDWorker
from agents.match_planner import MatchPlannerAgent
from agents.match_worker import MatchWorker
from agents.profile_from_resume import ProfileFromResumeAgent
from agents.profile_worker import ProfileWorker
from agents.resume_composer import ResumeComposerAgent
from agents.resume_composer_worker import ResumeComposerWorker
from core.config import get_default_model
from core.events import Event, InMemoryEventBus
from core.llm_client import OpenAIGPT5LLMClient
from core.llm_step_worker import LLMStepWorker
from core.pipeline_events import (
    COMPOSE_COMPLETED,
    COMPOSE_REQUESTED,
    JD_COMPLETED,
    JD_REQUESTED,
    MATCH_COMPLETED,
    MATCH_REQUESTED,
    PROFILE_COMPLETED,
    PROFILE_REQUESTED,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = ArgumentParser(description="Full event-driven resume pipeline demo.")
    parser.add_argument("jd_file", type=FileType("r"), help="Path to job description text file.")
    parser.add_argument("resume_file", type=FileType("r"), help="Path to resume text file.")
    args = parser.parse_args()

    jd_text = args.jd_file.read()
    resume_text = args.resume_file.read()

    bus = InMemoryEventBus()
    llm = OpenAIGPT5LLMClient()
    model_name = get_default_model()

    # Agents
    jd_agent = JDAnalysisAgent(llm=llm, model=model_name)
    profile_agent = ProfileFromResumeAgent(llm=llm, model=model_name)
    match_agent = MatchPlannerAgent(llm=llm, model=model_name)
    composer_agent = ResumeComposerAgent(llm=llm, model=model_name)

    # Workers
    llm_worker = LLMStepWorker(bus=bus, llm=llm, model=model_name)
    jd_worker = JDWorker(bus=bus, agent=jd_agent)
    profile_worker = ProfileWorker(bus=bus, agent=profile_agent)
    match_worker = MatchWorker(bus=bus, agent=match_agent)
    composer_worker = ResumeComposerWorker(bus=bus, agent=composer_agent)

    # Start long-running workers in background threads.
    threading.Thread(target=llm_worker.run_forever, daemon=True).start()
    threading.Thread(target=jd_worker.run_jd_requests, daemon=True).start()
    threading.Thread(target=jd_worker.run_llm_results, daemon=True).start()
    threading.Thread(target=profile_worker.run_profile_requests, daemon=True).start()
    threading.Thread(target=profile_worker.run_llm_results, daemon=True).start()
    threading.Thread(target=match_worker.run_match_requests, daemon=True).start()
    threading.Thread(target=match_worker.run_llm_results, daemon=True).start()
    threading.Thread(target=composer_worker.run_compose_requests, daemon=True).start()
    threading.Thread(target=composer_worker.run_llm_results, daemon=True).start()

    job_id = "full-demo-job-1"

    # 1) JD analysis
    logger.info("Publishing jd.requested for correlation_id=%s", job_id)
    bus.publish(
        Event(
            type=JD_REQUESTED,
            payload={"job_description": jd_text},
            correlation_id=job_id,
        )
    )

    logger.info("Waiting for jd.completed...")
    jd_payload = None
    for event in bus.subscribe(JD_COMPLETED):
        if event.correlation_id == job_id:
            jd_payload = event.payload["jd"]
            break

    # 2) Profile extraction
    logger.info("Publishing profile.requested for correlation_id=%s", job_id)
    bus.publish(
        Event(
            type=PROFILE_REQUESTED,
            payload={"resume_text": resume_text},
            correlation_id=job_id,
        )
    )

    logger.info("Waiting for profile.completed...")
    profile_payload = None
    for event in bus.subscribe(PROFILE_COMPLETED):
        if event.correlation_id == job_id:
            profile_payload = event.payload["profile"]
            break

    # 3) Match planning
    logger.info("Publishing match.requested for correlation_id=%s", job_id)
    bus.publish(
        Event(
            type=MATCH_REQUESTED,
            payload={"jd": jd_payload, "profile": profile_payload},
            correlation_id=job_id,
        )
    )

    logger.info("Waiting for match.completed...")
    plan_payload = None
    for event in bus.subscribe(MATCH_COMPLETED):
        if event.correlation_id == job_id:
            plan_payload = event.payload["plan"]
            break

    # 4) Resume composition
    logger.info("Publishing compose.requested for correlation_id=%s", job_id)
    bus.publish(
        Event(
            type=COMPOSE_REQUESTED,
            payload={"jd": jd_payload, "profile": profile_payload, "plan": plan_payload},
            correlation_id=job_id,
        )
    )

    logger.info("Waiting for compose.completed...")
    tailored_payload = None
    for event in bus.subscribe(COMPOSE_COMPLETED):
        if event.correlation_id == job_id:
            tailored_payload = event.payload["tailored"]
            break

    # Print final results.
    print("=== JDAnalysisResult ===")
    print(json.dumps(jd_payload, indent=2))
    print("\n=== ProfessionalProfile ===")
    print(json.dumps(profile_payload, indent=2))
    print("\n=== ResumePlan ===")
    print(json.dumps(plan_payload, indent=2))
    print("\n=== TailoredResume ===")
    print(json.dumps(tailored_payload, indent=2))


if __name__ == "__main__":
    main()
