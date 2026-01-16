"""Event-driven worker for ResumeComposerAgent.

Adapts ResumeComposerAgent to the event bus and central LLM step worker:

1. Listen for `compose.requested` events containing JD, profile, and plan.
2. Publish `llm_step.requested` with messages + COMPOSER_SCHEMA_TEXT,
   reply_to set to `compose.llm.completed`.
3. Listen for `compose.llm.completed` events carrying parsed JSON data.
4. Use ResumeComposerAgent.parse_result to obtain TailoredResume and
   publish `compose.completed`.
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.resume_composer import COMPOSER_SCHEMA_TEXT, ResumeComposerAgent
from core.events import Event, EventBus
from core.models import JDAnalysisResult, ProfessionalProfile, ResumePlan
from core.pipeline_events import (
    COMPOSE_COMPLETED,
    COMPOSE_LLM_COMPLETED,
    COMPOSE_REQUESTED,
    LLM_STEP_REQUESTED,
)


@dataclass(slots=True)
class ResumeComposerWorker:
    """Worker bridging resume composition events and the LLM step worker."""

    bus: EventBus
    agent: ResumeComposerAgent

    def run_compose_requests(self) -> None:
        """Consume compose.requested events and emit llm_step.requested."""
        for event in self.bus.subscribe(COMPOSE_REQUESTED):
            cid = event.correlation_id
            jd_data = event.payload.get("jd")
            profile_data = event.payload.get("profile")
            plan_data = event.payload.get("plan")

            jd = JDAnalysisResult.model_validate(jd_data)
            profile = ProfessionalProfile.model_validate(profile_data)
            plan = ResumePlan.model_validate(plan_data)

            messages = self.agent.build_messages(jd, profile, plan)
            payload = {
                "messages": messages,
                "schema_text": COMPOSER_SCHEMA_TEXT,
            }
            self.bus.publish(
                Event(
                    type=LLM_STEP_REQUESTED,
                    payload=payload,
                    correlation_id=cid,
                    reply_to=COMPOSE_LLM_COMPLETED,
                )
            )

    def run_llm_results(self) -> None:
        """Consume compose.llm.completed events and emit compose.completed."""
        for event in self.bus.subscribe(COMPOSE_LLM_COMPLETED):
            cid = event.correlation_id
            result = event.payload.get("result")
            if not isinstance(result, dict):
                raise ValueError("Resume composer worker expected a dict result payload")
            tailored = self.agent.parse_result(result)
            self.bus.publish(
                Event(
                    type=COMPOSE_COMPLETED,
                    payload={"tailored": tailored.model_dump()},
                    correlation_id=cid,
                )
            )
