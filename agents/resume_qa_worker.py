"""Event-driven worker for ResumeQAAgent.

Adapts ResumeQAAgent to the event bus and central LLM step worker:

1. Listen for `qa.requested` events containing JD, profile, and tailored resume.
2. Publish `llm_step.requested` with messages + QA_SCHEMA_TEXT,
   reply_to set to `qa.llm.completed`.
3. Listen for `qa.llm.completed` events carrying parsed JSON data.
4. Use ResumeQAAgent.parse_result to obtain ResumeQAResult and
   publish `qa.completed`.
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.qa_shared import QA_SCHEMA_TEXT
from agents.resume_qa import ResumeQAAgent
from core.events import Event, EventBus
from core.models import JDAnalysisResult, ProfessionalProfile, TailoredResume
from core.pipeline_events import LLM_STEP_REQUESTED, QA_COMPLETED, QA_LLM_COMPLETED, QA_REQUESTED


@dataclass(slots=True)
class ResumeQAWorker:
    """Worker bridging resume QA events and the LLM step worker."""

    bus: EventBus
    agent: ResumeQAAgent

    def run_qa_requests(self) -> None:
        """Consume qa.requested events and emit llm_step.requested."""
        for event in self.bus.subscribe(QA_REQUESTED):
            cid = event.correlation_id
            jd_data = event.payload.get("jd")
            profile_data = event.payload.get("profile")
            resume_data = event.payload.get("resume")

            jd = JDAnalysisResult.model_validate(jd_data)
            profile = ProfessionalProfile.model_validate(profile_data)
            resume = TailoredResume.model_validate(resume_data)

            messages = self.agent.build_messages(jd, profile, resume)
            payload = {
                "messages": messages,
                "schema_text": QA_SCHEMA_TEXT,
            }
            self.bus.publish(
                Event(
                    type=LLM_STEP_REQUESTED,
                    payload=payload,
                    correlation_id=cid,
                    reply_to=QA_LLM_COMPLETED,
                )
            )

    def run_llm_results(self) -> None:
        """Consume qa.llm.completed events and emit qa.completed."""
        for event in self.bus.subscribe(QA_LLM_COMPLETED):
            cid = event.correlation_id
            result = event.payload.get("result")
            if not isinstance(result, dict):
                raise ValueError("Resume QA worker expected a dict result payload")
            qa_result = self.agent.parse_result(result)
            self.bus.publish(
                Event(
                    type=QA_COMPLETED,
                    payload={"qa": qa_result.model_dump()},
                    correlation_id=cid,
                )
            )
