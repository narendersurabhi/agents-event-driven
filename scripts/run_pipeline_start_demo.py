"""Demo script: run full pipeline via pipeline.start event.

This uses PipelineOrchestrator plus all workers so that the caller only
publishes pipeline.start and waits for pipeline.completed.

Usage:
  python -m scripts.run_pipeline_start_demo jd.txt resume.txt
"""

from __future__ import annotations

from argparse import ArgumentParser, FileType
import json
import logging
import threading

from agents.jd_analysis import JDAnalysisAgent
from agents.jd_worker import JDWorker
from agents.match_planner import MatchPlannerAgent
from agents.match_worker import MatchWorker
from agents.profile_from_resume import ProfileFromResumeAgent
from agents.profile_worker import ProfileWorker
from agents.qa_improver import QAImproveAgent
from agents.qa_improver_worker import QAImproveWorker
from agents.resume_composer import ResumeComposerAgent
from agents.resume_composer_worker import ResumeComposerWorker
from agents.resume_qa import ResumeQAAgent
from agents.resume_qa_worker import ResumeQAWorker
from core.config import get_default_model
from core.events import Event, InMemoryEventBus
from core.llm_client import OpenAIGPT5LLMClient
from core.llm_step_worker import LLMStepWorker
from core.pipeline_orchestrator import (
    PIPELINE_COMPLETED,
    PIPELINE_START,
    PipelineOrchestrator,
)
from core.pipeline_store import InMemoryPipelineStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = ArgumentParser(description="Pipeline.start demo for full event-driven pipeline.")
    parser.add_argument("jd_file", type=FileType("r"), help="Path to job description text file.")
    parser.add_argument("resume_file", type=FileType("r"), help="Path to resume text file.")
    parser.add_argument("--no-qa", action="store_true", help="Skip QA step.")
    parser.add_argument("--no-improve", action="store_true", help="Skip QA improver step.")
    args = parser.parse_args()

    jd_text = args.jd_file.read()
    resume_text = args.resume_file.read()
    run_qa = not args.no_qa
    run_improver = not args.no_improve

    bus = InMemoryEventBus()
    llm = OpenAIGPT5LLMClient()
    model_name = get_default_model()
    store = InMemoryPipelineStore()

    # Agents
    jd_agent = JDAnalysisAgent(llm=llm, model=model_name)
    profile_agent = ProfileFromResumeAgent(llm=llm, model=model_name)
    match_agent = MatchPlannerAgent(llm=llm, model=model_name)
    composer_agent = ResumeComposerAgent(llm=llm, model=model_name)
    qa_agent = ResumeQAAgent(llm=llm, model=model_name)
    qa_improve_agent = QAImproveAgent(llm=llm, model=model_name)

    # Workers
    llm_worker = LLMStepWorker(bus=bus, llm=llm, model=model_name)
    jd_worker = JDWorker(bus=bus, agent=jd_agent)
    profile_worker = ProfileWorker(bus=bus, agent=profile_agent)
    match_worker = MatchWorker(bus=bus, agent=match_agent)
    composer_worker = ResumeComposerWorker(bus=bus, agent=composer_agent)
    qa_worker = ResumeQAWorker(bus=bus, agent=qa_agent)
    qa_improve_worker = QAImproveWorker(bus=bus, agent=qa_improve_agent)
    orchestrator = PipelineOrchestrator(bus=bus, store=store)

    # Start long-running workers.
    threading.Thread(target=llm_worker.run_forever, daemon=True).start()
    threading.Thread(target=jd_worker.run_jd_requests, daemon=True).start()
    threading.Thread(target=jd_worker.run_llm_results, daemon=True).start()
    threading.Thread(target=profile_worker.run_profile_requests, daemon=True).start()
    threading.Thread(target=profile_worker.run_llm_results, daemon=True).start()
    threading.Thread(target=match_worker.run_match_requests, daemon=True).start()
    threading.Thread(target=match_worker.run_llm_results, daemon=True).start()
    threading.Thread(target=composer_worker.run_compose_requests, daemon=True).start()
    threading.Thread(target=composer_worker.run_llm_results, daemon=True).start()
    threading.Thread(target=qa_worker.run_qa_requests, daemon=True).start()
    threading.Thread(target=qa_worker.run_llm_results, daemon=True).start()
    threading.Thread(target=qa_improve_worker.run_improve_requests, daemon=True).start()
    threading.Thread(target=qa_improve_worker.run_llm_results, daemon=True).start()

    # Start orchestrator loops.
    threading.Thread(target=orchestrator.run_pipeline_start, daemon=True).start()
    threading.Thread(target=orchestrator.run_jd_completed, daemon=True).start()
    threading.Thread(target=orchestrator.run_profile_completed, daemon=True).start()
    threading.Thread(target=orchestrator.run_match_completed, daemon=True).start()
    threading.Thread(target=orchestrator.run_compose_completed, daemon=True).start()
    threading.Thread(target=orchestrator.run_qa_completed, daemon=True).start()
    threading.Thread(target=orchestrator.run_qa_improve_completed, daemon=True).start()
    threading.Thread(target=orchestrator.run_pipeline_resume, daemon=True).start()

    job_id = "pipeline-start-demo-1"
    logger.info(
        "Publishing pipeline.start correlation_id=%s run_qa=%s run_improver=%s",
        job_id,
        run_qa,
        run_improver,
    )
    bus.publish(
        Event(
            type=PIPELINE_START,
            payload={
                "jd_text": jd_text,
                "resume_text": resume_text,
                "run_qa": run_qa,
                "run_improver": run_improver,
            },
            correlation_id=job_id,
        )
    )

    logger.info("Waiting for pipeline.completed (or resume-able failures)...")
    for event in bus.subscribe(PIPELINE_COMPLETED):
        if event.correlation_id == job_id:
            payload = event.payload
            print("=== JDAnalysisResult ===")
            print(json.dumps(payload["jd"], indent=2))
            print("\n=== ProfessionalProfile ===")
            print(json.dumps(payload["profile"], indent=2))
            print("\n=== ResumePlan ===")
            print(json.dumps(payload["plan"], indent=2))
            print("\n=== TailoredResume (final) ===")
            print(json.dumps(payload["improved"], indent=2))
            if payload.get("qa") is not None:
                print("\n=== ResumeQAResult ===")
                print(json.dumps(payload["qa"], indent=2))
            break


if __name__ == "__main__":
    main()
