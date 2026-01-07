"""Event-driven worker for CoverLetterAgent.

Flow:
1. Listen for `cover_letter.requested` events containing JD, profile, and final resume.
2. Publish `llm_step.requested` with messages + COVER_LETTER_SCHEMA_TEXT,
   reply_to set to `cover_letter.llm.completed`.
3. Listen for `cover_letter.llm.completed` events carrying parsed JSON data.
4. Use CoverLetterAgent.parse_result to obtain a CoverLetter and
   publish `cover_letter.completed`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agents.cover_letter_agent import COVER_LETTER_SCHEMA_TEXT, CoverLetterAgent
from core.events import Event, EventBus
from core.models import CoverLetter, JDAnalysisResult, ProfessionalProfile, TailoredResume
from core.pipeline_events import (
    COVER_LETTER_COMPLETED,
    COVER_LETTER_LLM_COMPLETED,
    COVER_LETTER_REQUESTED,
    LLM_STEP_REQUESTED,
)
from core.obs import JsonRepoLogger, Span, Logger


@dataclass(slots=True)
class CoverLetterWorker:
    """Worker bridging cover letter events and the LLM step worker."""

    bus: EventBus
    agent: CoverLetterAgent
    logger: Logger = field(default_factory=lambda: JsonRepoLogger(service="cover_letter"))

    def run_cover_letter_requests(self) -> None:
        """Consume cover_letter.requested events and emit llm_step.requested."""
        for event in self.bus.subscribe(COVER_LETTER_REQUESTED):
            cid = event.correlation_id
            with Span(self.logger, "cover_letter.request", {"cid": cid}):
                jd_data = event.payload.get("jd")
                profile_data = event.payload.get("profile")
                resume_data = event.payload.get("resume")

                jd = JDAnalysisResult.model_validate(jd_data)
                profile = ProfessionalProfile.model_validate(profile_data)
                resume = TailoredResume.model_validate(resume_data)

                messages = self.agent.build_messages(jd, profile, resume)
                payload = {
                    "messages": messages,
                    "schema_text": COVER_LETTER_SCHEMA_TEXT,
                }
                self.bus.publish(
                    Event(
                        type=LLM_STEP_REQUESTED,
                        payload=payload,
                        correlation_id=cid,
                        reply_to=COVER_LETTER_LLM_COMPLETED,
                    )
                )

    def run_llm_results(self) -> None:
        """Consume cover_letter.llm.completed events and emit cover_letter.completed."""
        for event in self.bus.subscribe(COVER_LETTER_LLM_COMPLETED):
            cid = event.correlation_id
            with Span(self.logger, "cover_letter.result", {"cid": cid}):
                result = event.payload.get("result")
                cover_letter: CoverLetter = self.agent.parse_result(result)
                self.bus.publish(
                    Event(
                        type=COVER_LETTER_COMPLETED,
                        payload={"cover_letter": cover_letter.model_dump()},
                        correlation_id=cid,
                    )
                )
