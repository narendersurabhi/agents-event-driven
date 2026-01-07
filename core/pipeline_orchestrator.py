"""Event-driven pipeline orchestrator.

Orchestrates the full resume pipeline via events only:

pipeline.start
  → jd.requested + profile.requested
  → jd.completed + profile.completed
  → match.requested → match.completed
  → compose.requested → compose.completed
  → qa.requested (optional) → qa.completed
  → qa_improve.requested (optional) → qa_improve.completed
  → pipeline.completed

Agents never call each other; this component only reacts to events and
publishes new ones based on per-job state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from core.events import Event, EventBus
from core.pipeline_events import (
    COMPOSE_COMPLETED,
    COMPOSE_REQUESTED,
    COVER_LETTER_COMPLETED,
    COVER_LETTER_REQUESTED,
    JD_COMPLETED,
    JD_REQUESTED,
    MATCH_COMPLETED,
    MATCH_REQUESTED,
    PIPELINE_COMPLETED,
    PIPELINE_RESTART_COMPOSE,
    PIPELINE_RESUME,
    PIPELINE_START,
    PROFILE_COMPLETED,
    PROFILE_REQUESTED,
    QA_COMPLETED,
    QA_IMPROVE_COMPLETED,
    QA_IMPROVE_REQUESTED,
    QA_REQUESTED,
)
from core.pipeline_store import PipelineStore


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineState:
    """In-memory per-job state tracked by the orchestrator."""

    jd: Optional[Dict[str, Any]] = None
    profile: Optional[Dict[str, Any]] = None
    plan: Optional[Dict[str, Any]] = None
    tailored: Optional[Dict[str, Any]] = None
    qa: Optional[Dict[str, Any]] = None
    cover_letter: Optional[Dict[str, Any]] = None
    run_qa: bool = True
    run_improver: bool = True
    stage: str = "PENDING"


@dataclass(slots=True)
class PipelineOrchestrator:
    """Orchestrates the full pipeline purely via events."""

    bus: EventBus
    store: PipelineStore
    logger: logging.Logger = field(default_factory=lambda: logger)
    _states: Dict[str, PipelineState] = field(default_factory=dict)

    def _state_for(self, cid: str) -> PipelineState:
        if cid not in self._states:
            # Try to hydrate from persistent store first.
            snapshot = self.store.load(cid)
            if snapshot is not None:
                self._states[cid] = self._state_from_snapshot(snapshot)
            else:
                self._states[cid] = PipelineState()
        return self._states[cid]

    def get_state_snapshot(self, cid: str) -> Optional[Dict[str, Any]]:
        """Return a shallow snapshot of the current state for a job."""
        state = self._states.get(cid)
        if state is None:
            return None
        return {
            "stage": state.stage,
            "run_qa": state.run_qa,
            "run_improver": state.run_improver,
            "jd": state.jd,
            "profile": state.profile,
            "plan": state.plan,
            "tailored": state.tailored,
            "qa": state.qa,
            "cover_letter": state.cover_letter,
        }

    @staticmethod
    def _state_from_snapshot(snapshot: Dict[str, Any]) -> PipelineState:
        """Rebuild a PipelineState from a stored snapshot dict."""
        return PipelineState(
            jd=snapshot.get("jd"),
            profile=snapshot.get("profile"),
            plan=snapshot.get("plan"),
            tailored=snapshot.get("tailored"),
            qa=snapshot.get("qa"),
            cover_letter=snapshot.get("cover_letter"),
            run_qa=bool(snapshot.get("run_qa", True)),
            run_improver=bool(snapshot.get("run_improver", True)),
            stage=str(snapshot.get("stage", "PENDING")),
        )

    def _persist(self, cid: str) -> None:
        """Persist the current state snapshot for cid, if available."""
        snap = self.get_state_snapshot(cid)
        if snap is not None:
            self.store.save(cid, snap)

    # ----- Entry point -----

    def run_pipeline_start(self) -> None:
        """Handle pipeline.start events and kick off jd/profile steps."""
        for event in self.bus.subscribe(PIPELINE_START):
            cid = event.correlation_id or ""
            if not cid:
                continue
            payload = event.payload
            jd_text: str = payload.get("jd_text", "")
            resume_text: str = payload.get("resume_text", "")
            run_qa: bool = bool(payload.get("run_qa", True))
            run_improver: bool = bool(payload.get("run_improver", True))
            force_profile_refresh: bool = bool(payload.get("force_profile_refresh", False))

            state = self._state_for(cid)
            state.run_qa = run_qa
            state.run_improver = run_improver
            state.stage = "STARTED"
            self._persist(cid)

            self.logger.info(
                "pipeline.start cid=%s run_qa=%s run_improver=%s force_profile_refresh=%s",
                cid,
                run_qa,
                run_improver,
                force_profile_refresh,
            )

            # Kick off JD and profile extraction in parallel.
            self.bus.publish(
                Event(
                    type=JD_REQUESTED,
                    payload={"job_description": jd_text},
                    correlation_id=cid,
                )
            )
            self.bus.publish(
                Event(
                    type=PROFILE_REQUESTED,
                    payload={
                        "resume_text": resume_text,
                        "force_refresh": force_profile_refresh,
                    },
                    correlation_id=cid,
                )
            )

    def run_pipeline_resume(self) -> None:
        """Handle pipeline.resume events and restart from the next pending step.

        This uses in-memory state only; it assumes the process is still running
        and the original PipelineState for the correlation_id is available.
        """
        for event in self.bus.subscribe(PIPELINE_RESUME):
            cid = event.correlation_id or ""
            if not cid:
                continue
            state = self._state_for(cid)
            self.logger.info("pipeline.resume cid=%s", cid)

            # Determine next step based on what we already have.
            if state.jd and state.profile and not state.plan:
                # JD + profile ready, but no plan: resume at match.
                self._publish_match_requested(cid, state)
            elif state.plan and not state.tailored:
                # Plan ready, but no tailored resume: resume at compose.
                self.logger.info("pipeline.compose_requested cid=%s", cid)
                self.bus.publish(
                    Event(
                        type=COMPOSE_REQUESTED,
                        payload={"jd": state.jd, "profile": state.profile, "plan": state.plan},
                        correlation_id=cid,
                    )
                )
            elif state.tailored and state.run_qa and not state.qa:
                # Tailored resume ready, QA requested but not done: resume at QA.
                self.logger.info("pipeline.qa_requested cid=%s", cid)
                self.bus.publish(
                    Event(
                        type=QA_REQUESTED,
                        payload={"jd": state.jd, "profile": state.profile, "resume": state.tailored},
                        correlation_id=cid,
                    )
                )
            elif state.qa and state.run_improver:
                # QA done, improver requested but not done: resume at QA improver.
                self.logger.info("pipeline.qa_improve_requested cid=%s", cid)
                self.bus.publish(
                    Event(
                        type=QA_IMPROVE_REQUESTED,
                        payload={
                            "jd": state.jd,
                            "profile": state.profile,
                            "resume": state.tailored,
                            "qa": state.qa,
                        },
                        correlation_id=cid,
                    )
                )
            # Otherwise nothing to resume (either not started or already completed).

    def run_pipeline_restart_compose(self) -> None:
        """Handle pipeline.restart_compose events and re-run compose + downstream steps.

        This assumes jd, profile, and plan already exist for the job. It clears the
        existing tailored/qa state (if any) and publishes a new COMPOSE_REQUESTED
        event so the pipeline can recompute the TailoredResume and then proceed
        through QA and QA improver as configured.
        """
        for event in self.bus.subscribe(PIPELINE_RESTART_COMPOSE):
            cid = event.correlation_id or ""
            if not cid:
                continue
            state = self._state_for(cid)
            if not (state.jd and state.profile and state.plan):
                self.logger.warning(
                    "pipeline.restart_compose.missing_prereqs cid=%s",
                    cid,
                )
                continue

            self.logger.info("pipeline.restart_compose cid=%s", cid)
            # Clear downstream artifacts so fresh ones are produced.
            state.tailored = None
            state.qa = None
            state.stage = "COMPOSE_RESTARTED"
            self._persist(cid)

            self.bus.publish(
                Event(
                    type=COMPOSE_REQUESTED,
                    payload={"jd": state.jd, "profile": state.profile, "plan": state.plan},
                    correlation_id=cid,
                )
            )

    # ----- Intermediate steps -----

    def run_jd_completed(self) -> None:
        """React to jd.completed and trigger match when possible."""
        for event in self.bus.subscribe(JD_COMPLETED):
            cid = event.correlation_id or ""
            if not cid:
                continue
            state = self._state_for(cid)
            state.jd = event.payload.get("jd")
            state.stage = "JD_COMPLETED"
            self._persist(cid)
            self.logger.info("pipeline.jd_completed cid=%s", cid)

            if state.profile and not state.plan:
                self._publish_match_requested(cid, state)

    def run_profile_completed(self) -> None:
        """React to profile.completed and trigger match when possible."""
        for event in self.bus.subscribe(PROFILE_COMPLETED):
            cid = event.correlation_id or ""
            if not cid:
                continue
            state = self._state_for(cid)
            state.profile = event.payload.get("profile")
            state.stage = "PROFILE_COMPLETED"
            self._persist(cid)
            self.logger.info("pipeline.profile_completed cid=%s", cid)

            if state.jd and not state.plan:
                self._publish_match_requested(cid, state)

    def _publish_match_requested(self, cid: str, state: PipelineState) -> None:
        self.logger.info("pipeline.match_requested cid=%s", cid)
        self.bus.publish(
            Event(
                type=MATCH_REQUESTED,
                payload={"jd": state.jd, "profile": state.profile},
                correlation_id=cid,
            )
        )

    def run_match_completed(self) -> None:
        """React to match.completed and trigger compose."""
        for event in self.bus.subscribe(MATCH_COMPLETED):
            cid = event.correlation_id or ""
            if not cid:
                continue
            state = self._state_for(cid)
            state.plan = event.payload.get("plan")
            state.stage = "MATCH_COMPLETED"
            self._persist(cid)
            self.logger.info("pipeline.match_completed cid=%s", cid)

            if state.jd and state.profile and state.plan:
                self.logger.info("pipeline.compose_requested cid=%s", cid)
                self.bus.publish(
                    Event(
                        type=COMPOSE_REQUESTED,
                        payload={"jd": state.jd, "profile": state.profile, "plan": state.plan},
                        correlation_id=cid,
                    )
                )

    def run_compose_completed(self) -> None:
        """React to compose.completed and trigger QA or finish."""
        for event in self.bus.subscribe(COMPOSE_COMPLETED):
            cid = event.correlation_id or ""
            if not cid:
                continue
            state = self._state_for(cid)
            state.tailored = event.payload.get("tailored")
            state.stage = "COMPOSE_COMPLETED"
            self._persist(cid)
            self.logger.info("pipeline.compose_completed cid=%s", cid)

            if state.run_qa:
                self.logger.info("pipeline.qa_requested cid=%s", cid)
                self.bus.publish(
                    Event(
                        type=QA_REQUESTED,
                        payload={"jd": state.jd, "profile": state.profile, "resume": state.tailored},
                        correlation_id=cid,
                    )
                )
            else:
                self._publish_pipeline_completed(cid, state, improved=None)

    def run_qa_completed(self) -> None:
        """React to qa.completed and trigger QA improver or finish."""
        for event in self.bus.subscribe(QA_COMPLETED):
            cid = event.correlation_id or ""
            if not cid:
                continue
            state = self._state_for(cid)
            state.qa = event.payload.get("qa")
            state.stage = "QA_COMPLETED"
            self._persist(cid)
            self.logger.info("pipeline.qa_completed cid=%s", cid)

            if state.run_improver:
                self.logger.info("pipeline.qa_improve_requested cid=%s", cid)
                self.bus.publish(
                    Event(
                        type=QA_IMPROVE_REQUESTED,
                        payload={
                            "jd": state.jd,
                            "profile": state.profile,
                            "resume": state.tailored,
                            "qa": state.qa,
                        },
                        correlation_id=cid,
                    )
                )
            else:
                self._publish_pipeline_completed(cid, state, improved=None)

    def run_qa_improve_completed(self) -> None:
        """React to qa_improve.completed and finish the pipeline."""
        for event in self.bus.subscribe(QA_IMPROVE_COMPLETED):
            cid = event.correlation_id or ""
            if not cid:
                continue
            state = self._state_for(cid)
            improved = event.payload.get("tailored")
            state.stage = "QA_IMPROVE_COMPLETED"
            self._persist(cid)
            self.logger.info("pipeline.qa_improve_completed cid=%s", cid)
            # After we have an improved resume, request a cover letter.
            final_resume = improved or state.tailored
            if final_resume is not None:
                self.bus.publish(
                    Event(
                        type=COVER_LETTER_REQUESTED,
                        payload={
                            "jd": state.jd,
                            "profile": state.profile,
                            "resume": final_resume,
                        },
                        correlation_id=cid,
                    )
                )
            else:
                # Fallback: complete pipeline even if we cannot generate a cover letter.
                self._publish_pipeline_completed(cid, state, improved=improved)

    def run_cover_letter_completed(self) -> None:
        """React to cover_letter.completed and finish the pipeline."""
        for event in self.bus.subscribe(COVER_LETTER_COMPLETED):
            cid = event.correlation_id or ""
            if not cid:
                continue
            state = self._state_for(cid)
            state.cover_letter = event.payload.get("cover_letter")
            state.stage = "COVER_LETTER_COMPLETED"
            self._persist(cid)
            self.logger.info("pipeline.cover_letter_completed cid=%s", cid)
            self._publish_pipeline_completed(cid, state, improved=None)

    # ----- Finalization -----

    def _publish_pipeline_completed(
        self,
        cid: str,
        state: PipelineState,
        improved: Optional[Dict[str, Any]],
    ) -> None:
        """Publish pipeline.completed with all available artifacts."""
        state.stage = "COMPLETED"
        self._persist(cid)
        payload: Dict[str, Any] = {
            "jd": state.jd,
            "profile": state.profile,
            "plan": state.plan,
            "tailored": state.tailored,
            "qa": state.qa,
            "improved": improved or state.tailored,
            "cover_letter": state.cover_letter,
        }
        self.logger.info("pipeline.completed cid=%s", cid)
        self.bus.publish(
            Event(
                type=PIPELINE_COMPLETED,
                payload=payload,
                correlation_id=cid,
            )
        )
