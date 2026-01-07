"""Central LLM step worker that owns JSON repair.

Pattern B: agents describe their LLM call (messages + schema text) via an
`llm_step.requested` event. This worker:

1. Calls the underlying LLM client.
2. Attempts to parse JSON from the raw response.
3. On failure, calls JsonRepairAgent with the provided schema_text.
4. Publishes either a completed event with parsed JSON, or a failure event.

Domain agents remain unaware of JsonRepairAgent and only consume the
structured JSON payload from the `llm_step.completed` events.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from core.events import Event, EventBus
from core.json_utils import parse_json_object
from core.json_repair import JsonRepairAgent
from core.llm_client import LLMClient
from core.obs import JsonRepoLogger, Logger, NullLogger
from core.pipeline_events import LLM_STEP_COMPLETED, LLM_STEP_FAILED, LLM_STEP_REQUESTED


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LLMStepWorker:
    """Worker that processes llm_step.requested events on an EventBus."""

    bus: EventBus
    llm: LLMClient
    model: str
    logger: logging.Logger = field(default_factory=lambda: logger)
    obs: Logger = field(default_factory=lambda: JsonRepoLogger(service="llm_step"))

    @staticmethod
    def _infer_step(reply_type: str) -> str:
        if ".llm." in reply_type:
            return reply_type.split(".llm.", 1)[0]
        if reply_type.endswith(".completed"):
            return reply_type.rsplit(".", 1)[0]
        return reply_type

    def run_forever(self) -> None:
        """Block and process LLM step requests indefinitely."""
        for event in self.bus.subscribe(LLM_STEP_REQUESTED):
            self._handle_request(event)

    def _handle_request(self, event: Event) -> None:
        cid = event.correlation_id
        payload = event.payload
        start_ns = time.time_ns()
        try:
            messages = payload["messages"]
            schema_text = payload.get("schema_text", "")
        except KeyError as exc:
            # NOTE: `self.logger` is the stdlib logger; it doesn't accept arbitrary kwargs.
            self.logger.error(
                "llm_step.invalid_event cid=%s missing=%s payload_keys=%s",
                cid,
                str(exc),
                list(payload.keys()),
            )
            self.bus.publish(
                Event(
                    type=LLM_STEP_FAILED,
                    payload={"error": f"Missing field: {exc}"},
                    correlation_id=cid,
                )
            )
            return

        reply_type = event.reply_to or LLM_STEP_COMPLETED
        step = self._infer_step(reply_type)
        # Best-effort observability. Never let metrics logging break the pipeline.
        obs = self.obs or NullLogger()
        obs.info(
            "llm_step.start",
            cid=cid,
            step=step,
            reply_type=reply_type,
            model=self.model,
        )

        try:
            repaired = False
            if cid:
                raw = self.llm.chat(messages=messages, model=self.model, temperature=0.0, req_id=cid)
            else:
                raw = self.llm.chat(messages=messages, model=self.model, temperature=0.0)
            try:
                data = parse_json_object(raw, ValueError)
            except ValueError as parse_err:
                self.logger.warning(
                    "llm_step.parse_failed cid=%s error=%s",
                    cid,
                    str(parse_err),
                )
                obs.warn(
                    "llm_step.parse_failed",
                    cid=cid,
                    step=step,
                    reply_type=reply_type,
                    model=self.model,
                    error=str(parse_err),
                )
                repair_agent = JsonRepairAgent(llm=self.llm, model=self.model)
                repaired_raw = repair_agent.repair(
                    raw,
                    schema_text=schema_text,
                    error=str(parse_err),
                    req_id=cid,
                )
                repaired = True
                data = parse_json_object(repaired_raw, ValueError)

            self.bus.publish(
                Event(
                    type=reply_type,
                    payload={"result": data},
                    correlation_id=cid,
                )
            )
            dur_ms = (time.time_ns() - start_ns) / 1e6
            obs.info(
                "llm_step.end",
                cid=cid,
                step=step,
                reply_type=reply_type,
                model=self.model,
                duration_ms=dur_ms,
                repaired=repaired,
            )
        except Exception as exc:
            self.logger.exception("llm_step.failed cid=%s error=%s", cid, str(exc))
            dur_ms = (time.time_ns() - start_ns) / 1e6
            obs.error(
                "llm_step.error",
                cid=cid,
                step=step,
                reply_type=reply_type,
                model=self.model,
                duration_ms=dur_ms,
                error=str(exc),
            )
            self.bus.publish(
                Event(
                    type=LLM_STEP_FAILED,
                    payload={"error": str(exc)},
                    correlation_id=cid,
                )
            )
