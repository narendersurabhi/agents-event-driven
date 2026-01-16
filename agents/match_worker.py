"""Event-driven worker for MatchPlannerAgent.

This worker adapts MatchPlannerAgent to the event bus and central LLM
step worker, following the same pattern as JDWorker and ProfileWorker:

1. Listen for `match.requested` events containing JD and profile payloads.
2. Publish `llm_step.requested` with messages + MATCH_PLAN_SCHEMA_TEXT,
   reply_to set to `match.llm.completed`.
3. Listen for `match.llm.completed` events carrying parsed JSON data.
4. Use MatchPlannerAgent.parse_result to obtain a ResumePlan and
   publish `match.completed`.
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.match_planner import MATCH_PLAN_SCHEMA_TEXT, MatchPlannerAgent
from core.events import Event, EventBus
from core.models import JDAnalysisResult, ProfessionalProfile
from core.pipeline_events import (
    LLM_STEP_REQUESTED,
    MATCH_COMPLETED,
    MATCH_LLM_COMPLETED,
    MATCH_REQUESTED,
)


@dataclass(slots=True)
class MatchWorker:
    """Worker bridging match planning events and the LLM step worker."""

    bus: EventBus
    agent: MatchPlannerAgent

    def run_match_requests(self) -> None:
        """Consume match.requested events and emit llm_step.requested."""
        for event in self.bus.subscribe(MATCH_REQUESTED):
            cid = event.correlation_id
            jd_data = event.payload.get("jd")
            profile_data = event.payload.get("profile")

            jd = JDAnalysisResult.model_validate(jd_data)
            profile = ProfessionalProfile.model_validate(profile_data)

            messages = self.agent.build_messages(jd, profile)
            payload = {
                "messages": messages,
                "schema_text": MATCH_PLAN_SCHEMA_TEXT,
            }
            self.bus.publish(
                Event(
                    type=LLM_STEP_REQUESTED,
                    payload=payload,
                    correlation_id=cid,
                    reply_to=MATCH_LLM_COMPLETED,
                )
            )

    def run_llm_results(self) -> None:
        """Consume match.llm.completed events and emit match.completed."""
        for event in self.bus.subscribe(MATCH_LLM_COMPLETED):
            cid = event.correlation_id
            result = event.payload.get("result")
            if not isinstance(result, dict):
                raise ValueError("Match worker expected a dict result payload")
            plan = self.agent.parse_result(result)
            self.bus.publish(
                Event(
                    type=MATCH_COMPLETED,
                    payload={"plan": plan.model_dump()},
                    correlation_id=cid,
                )
            )
