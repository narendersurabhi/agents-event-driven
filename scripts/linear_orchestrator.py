"""Direct (non-event-driven) resume pipeline orchestrator.

This module wires agents together in-process, without the EventBus/worker pattern.
It lives under `scripts/` (not `core/`) because it imports domain agents directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import anyio

from agents.jd_analysis_async import AsyncJDAnalysisAgent
from agents.match_planner_async import AsyncMatchPlannerAgent
from agents.profile_from_resume import ProfileFromResumeAgent
from agents.qa_improver_async import QAImproveAgent
from agents.resume_composer_async import AsyncResumeComposerAgent
from agents.resume_qa_async import AsyncResumeQAAgent, ResumeQAResult
from core.llm_client import AsyncLLMClient, LLMClient
from core.models import JDAnalysisResult, ProfessionalProfile, ResumePlan, TailoredResume
from core.obs import Logger, NullLogger
from core.state_machine import SimpleStateMachine, StateMachineBackend


@dataclass
class OrchestrationResult:
    jd: JDAnalysisResult
    profile: ProfessionalProfile
    plan: ResumePlan
    tailored: TailoredResume
    qa: ResumeQAResult | None = None
    improved: TailoredResume | None = None


class ResumePipelineOrchestrator:
    """Wires all agents together behind a pluggable state machine backend."""

    def __init__(
        self,
        async_llm: AsyncLLMClient,
        sync_llm: LLMClient,
        logger: Logger | None = None,
        backend: StateMachineBackend | None = None,
        run_qa: bool = True,
        run_improver: bool = True,
    ) -> None:
        self.logger = logger or NullLogger()
        self.sm: StateMachineBackend = backend or SimpleStateMachine()
        self.run_qa = run_qa
        self.run_improver = run_improver

        # Agents
        self.jd_agent = AsyncJDAnalysisAgent(llm=async_llm)
        self.profile_agent = ProfileFromResumeAgent(llm=sync_llm)
        self.plan_agent = AsyncMatchPlannerAgent(llm=async_llm)
        self.compose_agent = AsyncResumeComposerAgent(llm=async_llm)
        self.qa_agent = AsyncResumeQAAgent(llm=async_llm, logger=self.logger)
        self.improver = QAImproveAgent(llm=async_llm, logger=self.logger)

        self._setup_states()

    def _setup_states(self) -> None:
        states = [
            "PENDING",
            "JD_ANALYZED",
            "PROFILE_EXTRACTED",
            "PLAN_READY",
            "COMPOSED",
            "QA_DONE",
            "IMPROVED",
            "DONE",
            "FAILED",
        ]
        for s in states:
            self.sm.add_state(s)
        # Transitions chain; state updates happen in run() via triggers.
        chain = [
            ("analyze", "PENDING", "JD_ANALYZED"),
            ("profile", "JD_ANALYZED", "PROFILE_EXTRACTED"),
            ("plan", "PROFILE_EXTRACTED", "PLAN_READY"),
            ("compose", "PLAN_READY", "COMPOSED"),
            ("qa", "COMPOSED", "QA_DONE"),
            ("improve", "QA_DONE", "IMPROVED"),
            ("finish", "IMPROVED", "DONE"),
            # Allow short-circuits when QA or improver are skipped
            ("finish", "QA_DONE", "DONE"),
            ("finish", "COMPOSED", "DONE"),
        ]
        for trigger, src, dst in chain:
            self.sm.add_transition(trigger, src, dst)
        self.sm.set_state("PENDING")

    async def run(self, jd_text: str, resume_text: str) -> OrchestrationResult:
        try:
            # JD
            jd = await self.jd_agent.analyze(jd_text)
            self.sm.trigger("analyze")

            # Profile (sync agent; run in thread)
            profile = await anyio.to_thread.run_sync(self.profile_agent.extract, resume_text)
            self.sm.trigger("profile")

            # Plan
            plan = await self.plan_agent.plan(jd, profile)
            self.sm.trigger("plan")

            # Compose
            tailored = await self.compose_agent.compose(jd, profile, plan)
            self.sm.trigger("compose")

            qa_result: ResumeQAResult | None = None
            improved: TailoredResume | None = None

            if self.run_qa:
                qa_result = await self.qa_agent.review(jd, profile, tailored)
                self.sm.trigger("qa")

                if self.run_improver:
                    improved = await self.improver.improve(jd, profile, tailored, qa_result)
                    self.sm.trigger("improve")
                else:
                    improved = tailored
                    self.sm.trigger("finish")
            else:
                self.sm.trigger("finish")

            return OrchestrationResult(
                jd=jd,
                profile=profile,
                plan=plan,
                tailored=tailored,
                qa=qa_result,
                improved=improved,
            )
        except Exception as exc:
            self.logger.error("orchestration.failed", error=str(exc), state=self.sm.state)
            try:
                self.sm.set_state("FAILED")
            except Exception:
                pass
            raise


__all__ = ["OrchestrationResult", "ResumePipelineOrchestrator"]
