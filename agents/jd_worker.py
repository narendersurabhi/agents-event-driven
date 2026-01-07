"""Event-driven worker for JDAnalysisAgent.

This worker adapts JDAnalysisAgent to the event bus and central LLM
step worker. It follows the same pattern as ProfileWorker:

1. Listen for `jd.requested` events with a `job_description` payload.
2. Publish `llm_step.requested` describing the LLM call, with schema_text
   and reply_to set to `jd.llm.completed`.
3. Listen for `jd.llm.completed` events carrying parsed JSON data.
4. Use JDAnalysisAgent.parse_result to obtain JDAnalysisResult and
   publish `jd.completed`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.jd_analysis import JDAnalysisAgent, JD_SCHEMA_TEXT
from core.events import Event, EventBus
from core.pipeline_events import JD_COMPLETED, JD_LLM_COMPLETED, JD_REQUESTED, LLM_STEP_REQUESTED

@dataclass(slots=True)
class JDWorker:
    """Worker that bridges JD analysis events and the LLM step worker."""

    bus: EventBus
    agent: JDAnalysisAgent

    def run_jd_requests(self) -> None:
        """Consume jd.requested events and emit llm_step.requested."""
        for event in self.bus.subscribe(JD_REQUESTED):
            cid = event.correlation_id
            jd_text = event.payload.get("job_description", "")
            messages = self.agent.build_messages(jd_text)
            payload = {
                "messages": messages,
                "schema_text": JD_SCHEMA_TEXT,
            }
            self.bus.publish(
                Event(
                    type=LLM_STEP_REQUESTED,
                    payload=payload,
                    correlation_id=cid,
                    reply_to=JD_LLM_COMPLETED,
                )
            )

    def run_llm_results(self) -> None:
        """Consume jd.llm.completed events and emit jd.completed."""
        for event in self.bus.subscribe(JD_LLM_COMPLETED):
            cid = event.correlation_id
            result = event.payload.get("result")
            jd_result = self.agent.parse_result(result)
            self.bus.publish(
                Event(
                    type=JD_COMPLETED,
                    payload={"jd": jd_result.model_dump()},
                    correlation_id=cid,
                )
            )
