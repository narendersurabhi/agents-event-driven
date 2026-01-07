"""Event-driven worker for ProfileFromResumeAgent.

This worker is a thin adapter between the event bus and the profile
extraction agent. It does not call the LLM directly; instead it:

1. Listens for `profile.requested` events containing raw resume text.
2. Publishes an `llm_step.requested` event describing the LLM call
   (messages + schema_text) with reply_to set to `profile.llm.completed`.
3. Listens for `profile.llm.completed` events carrying parsed JSON
   from the central LLMStepWorker.
4. Uses ProfileFromResumeAgent.parse_result to turn that JSON into a
   ProfessionalProfile and publishes `profile.completed`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Dict
import json

from core.events import Event, EventBus
from agents.profile_from_resume import PROFILE_SCHEMA_TEXT, ProfileFromResumeAgent
from core.pipeline_events import (
    LLM_STEP_REQUESTED,
    PROFILE_COMPLETED,
    PROFILE_LLM_COMPLETED,
    PROFILE_REQUESTED,
)

@dataclass(slots=True)
class ProfileWorker:
    """Worker that bridges profile events and the central LLM step worker."""

    bus: EventBus
    agent: ProfileFromResumeAgent
    # Simple in-memory cache keyed by a hash of resume_text.
    _cache: Dict[str, dict] = field(default_factory=dict)
    # Map correlation_id -> cache key for in-flight requests.
    _cache_keys: Dict[str, str] = field(default_factory=dict)
    # Optional on-disk cache directory for persistence across restarts.
    _cache_dir: Path = field(default_factory=lambda: Path("profile_cache"))

    def run_profile_requests(self) -> None:
        """Consume profile.requested events and emit llm_step.requested.

        Each incoming event payload must contain a `resume_text` field.
        The correlation_id is propagated so downstream consumers can tie
        together the full pipeline for a given job.
        """
        for event in self.bus.subscribe(PROFILE_REQUESTED):
            cid = event.correlation_id
            resume_text = event.payload.get("resume_text", "") or ""
            force_refresh = bool(event.payload.get("force_refresh", False))

            # Compute a stable key for this resume text.
            key = sha256(resume_text.encode("utf-8")).hexdigest()

            # Cache hit: emit PROFILE_COMPLETED directly without calling the LLM,
            # unless the caller explicitly requested a refresh.
            cached = None if force_refresh else self._cache.get(key)
            # If not in memory and refresh not forced, try on-disk cache.
            if cached is None and not force_refresh and self._cache_dir:
                path = self._cache_dir / f"{key}.json"
                if path.exists():
                    try:
                        cached = json.loads(path.read_text(encoding="utf-8"))
                        self._cache[key] = cached
                    except json.JSONDecodeError:
                        cached = None

            if cached is not None:
                self.bus.publish(
                    Event(
                        type=PROFILE_COMPLETED,
                        payload={"profile": cached},
                        correlation_id=cid,
                    )
                )
                continue

            # Cache miss: go through the LLM step and remember the key for this cid.
            self._cache_keys[cid] = key
            messages = self.agent.build_messages(resume_text)
            llm_payload = {
                "messages": messages,
                "schema_text": PROFILE_SCHEMA_TEXT,
            }
            self.bus.publish(
                Event(
                    type=LLM_STEP_REQUESTED,
                    payload=llm_payload,
                    correlation_id=cid,
                    reply_to=PROFILE_LLM_COMPLETED,
                )
            )

    def run_llm_results(self) -> None:
        """Consume profile.llm.completed events and emit profile.completed.

        Expects each event payload to contain a `result` field with the
        parsed JSON object produced by LLMStepWorker.
        """
        for event in self.bus.subscribe(PROFILE_LLM_COMPLETED):
            cid = event.correlation_id
            result = event.payload.get("result")
            profile = self.agent.parse_result(result)
            profile_data = profile.model_dump()

            # Fill the cache for future jobs using the same resume text.
            key = self._cache_keys.pop(cid, None)
            if key is not None:
                self._cache[key] = profile_data
                if self._cache_dir:
                    self._cache_dir.mkdir(parents=True, exist_ok=True)
                    path = self._cache_dir / f"{key}.json"
                    tmp = path.with_suffix(".json.tmp")
                    tmp.write_text(
                        json.dumps(profile_data, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    tmp.replace(path)

            self.bus.publish(
                Event(
                    type=PROFILE_COMPLETED,
                    payload={"profile": profile_data},
                    correlation_id=cid,
                )
            )
