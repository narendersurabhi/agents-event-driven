"""Demo script: event-driven profile extraction pipeline.

Wires together:
- InMemoryEventBus
- LLMStepWorker (central LLM + JSON repair)
- ProfileFromResumeAgent + ProfileWorker

Usage:
  python -m scripts.run_profile_event_demo path/to/resume.txt
"""

from __future__ import annotations

import logging
import threading
from argparse import ArgumentParser, FileType

from agents.profile_from_resume import ProfileFromResumeAgent
from agents.profile_worker import ProfileWorker
from core.config import get_default_model
from core.events import Event, InMemoryEventBus
from core.llm_client import OpenAIGPT5LLMClient
from core.llm_step_worker import LLMStepWorker
from core.pipeline_events import PROFILE_COMPLETED, PROFILE_REQUESTED


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = ArgumentParser(description="Event-driven demo for ProfileFromResumeAgent.")
    parser.add_argument(
        "file",
        type=FileType("r"),
        help="Path to a text file containing the resume.",
    )
    args = parser.parse_args()

    resume_text = args.file.read()
    bus = InMemoryEventBus()

    llm = OpenAIGPT5LLMClient()
    model_name = get_default_model()

    profile_agent = ProfileFromResumeAgent(llm=llm, model=model_name)
    llm_worker = LLMStepWorker(bus=bus, llm=llm, model=model_name)
    profile_worker = ProfileWorker(bus=bus, agent=profile_agent)

    # Start workers in background threads.
    threading.Thread(target=llm_worker.run_forever, daemon=True).start()
    threading.Thread(target=profile_worker.run_profile_requests, daemon=True).start()
    threading.Thread(target=profile_worker.run_llm_results, daemon=True).start()

    job_id = "demo-job-1"
    logger.info("Publishing profile.requested event with correlation_id=%s", job_id)
    bus.publish(
        Event(
            type=PROFILE_REQUESTED,
            payload={"resume_text": resume_text},
            correlation_id=job_id,
        )
    )

    # Wait for profile.completed for this job_id.
    logger.info("Waiting for profile.completed event...")
    for event in bus.subscribe(PROFILE_COMPLETED):
        if event.correlation_id == job_id:
            profile = event.payload["profile"]
            print("ProfessionalProfile JSON:")
            # Pretty-print as JSON-like structure.
            import json

            print(json.dumps(profile, indent=2))
            break


if __name__ == "__main__":
    main()
