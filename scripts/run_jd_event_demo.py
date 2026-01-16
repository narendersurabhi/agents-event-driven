"""Demo script: event-driven JD analysis pipeline.

Wires together:
- InMemoryEventBus
- LLMStepWorker (central LLM + JSON repair)
- JDAnalysisAgent + JDWorker

Usage:
  python -m scripts.run_jd_event_demo path/to/jd.txt
"""

from __future__ import annotations

from argparse import ArgumentParser, FileType
import json
import logging
import threading

from agents.jd_analysis import JDAnalysisAgent
from agents.jd_worker import JDWorker
from core.config import get_default_model
from core.events import Event, InMemoryEventBus
from core.llm_client import OpenAIGPT5LLMClient
from core.llm_step_worker import LLMStepWorker
from core.pipeline_events import JD_COMPLETED, JD_REQUESTED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = ArgumentParser(description="Event-driven demo for JDAnalysisAgent.")
    parser.add_argument(
        "file",
        type=FileType("r"),
        help="Path to a text file containing the job description.",
    )
    args = parser.parse_args()

    jd_text = args.file.read()

    bus = InMemoryEventBus()
    llm = OpenAIGPT5LLMClient()
    model_name = get_default_model()

    jd_agent = JDAnalysisAgent(llm=llm, model=model_name)
    llm_worker = LLMStepWorker(bus=bus, llm=llm, model=model_name)
    jd_worker = JDWorker(bus=bus, agent=jd_agent)

    # Start workers in background threads.
    threading.Thread(target=llm_worker.run_forever, daemon=True).start()
    threading.Thread(target=jd_worker.run_jd_requests, daemon=True).start()
    threading.Thread(target=jd_worker.run_llm_results, daemon=True).start()

    job_id = "jd-demo-job-1"
    logger.info("Publishing jd.requested event with correlation_id=%s", job_id)
    bus.publish(
        Event(
            type=JD_REQUESTED,
            payload={"job_description": jd_text},
            correlation_id=job_id,
        )
    )

    # Wait for jd.completed for this job_id.
    logger.info("Waiting for jd.completed event...")
    for event in bus.subscribe(JD_COMPLETED):
        if event.correlation_id == job_id:
            jd_payload = event.payload["jd"]
            print("JDAnalysisResult JSON:")
            print(json.dumps(jd_payload, indent=2))
            break


if __name__ == "__main__":
    main()
