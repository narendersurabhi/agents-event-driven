"""Event-driven worker for QAImproveAgent.

Adapts QAImproveAgent to the event bus and central LLM step worker:

1. Listen for `qa_improve.requested` events containing JD, profile, resume, and QA result.
2. Publish `llm_step.requested` with messages + QA_IMPROVE_SCHEMA_TEXT,
   reply_to set to `qa_improve.llm.completed`.
3. Listen for `qa_improve.llm.completed` events carrying parsed JSON data.
4. Use QAImproveAgent.parse_result to obtain improved TailoredResume and
   publish `qa_improve.completed`.
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.qa_improver import QA_IMPROVE_SCHEMA_TEXT, QAImproveAgent
from agents.qa_shared import ResumeQAResult
from core.events import Event, EventBus
from core.models import JDAnalysisResult, ProfessionalProfile, TailoredResume
from core.pipeline_events import (
    LLM_STEP_REQUESTED,
    QA_IMPROVE_COMPLETED,
    QA_IMPROVE_LLM_COMPLETED,
    QA_IMPROVE_REQUESTED,
)


@dataclass(slots=True)
class QAImproveWorker:
    """Worker bridging QA improvement events and the LLM step worker."""

    bus: EventBus
    agent: QAImproveAgent

    def run_improve_requests(self) -> None:
        """Consume qa_improve.requested events and emit llm_step.requested."""
        for event in self.bus.subscribe(QA_IMPROVE_REQUESTED):
            cid = event.correlation_id
            jd_data = event.payload.get("jd")
            profile_data = event.payload.get("profile")
            resume_data = event.payload.get("resume")
            qa_data = event.payload.get("qa")

            jd = JDAnalysisResult.model_validate(jd_data)
            profile = ProfessionalProfile.model_validate(profile_data)
            resume = TailoredResume.model_validate(resume_data)
            qa = ResumeQAResult.model_validate(qa_data)

            messages = self.agent.build_messages(jd, profile, resume, qa)
            payload = {
                "messages": messages,
                "schema_text": QA_IMPROVE_SCHEMA_TEXT,
            }
            self.bus.publish(
                Event(
                    type=LLM_STEP_REQUESTED,
                    payload=payload,
                    correlation_id=cid,
                    reply_to=QA_IMPROVE_LLM_COMPLETED,
                )
            )

    def run_llm_results(self) -> None:
        """Consume qa_improve.llm.completed events and emit qa_improve.completed."""
        for event in self.bus.subscribe(QA_IMPROVE_LLM_COMPLETED):
            cid = event.correlation_id
            result = event.payload.get("result")
            improved = self.agent.parse_result(result)
            self.bus.publish(
                Event(
                    type=QA_IMPROVE_COMPLETED,
                    payload={"tailored": improved.model_dump()},
                    correlation_id=cid,
                )
            )
