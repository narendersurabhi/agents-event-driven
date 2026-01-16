"""Event name contracts for the event-driven resume pipeline.

This module centralizes event type strings so:
- `core/*` does not need to import `agents/*` just for constants.
- workers/orchestrators/scripts share a single source of truth.

Only constants live here (no side effects).
"""

from __future__ import annotations

# ----- Central LLM + JSON repair -----

LLM_STEP_REQUESTED = "llm_step.requested"
LLM_STEP_COMPLETED = "llm_step.completed"
LLM_STEP_FAILED = "llm_step.failed"

# ----- Pipeline lifecycle -----

PIPELINE_START = "pipeline.start"
PIPELINE_RESUME = "pipeline.resume"
PIPELINE_RESTART_COMPOSE = "pipeline.restart_compose"
PIPELINE_COMPLETED = "pipeline.completed"

# ----- JD analysis -----

JD_REQUESTED = "jd.requested"
JD_LLM_COMPLETED = "jd.llm.completed"
JD_COMPLETED = "jd.completed"

# ----- Profile extraction -----

PROFILE_REQUESTED = "profile.requested"
PROFILE_LLM_COMPLETED = "profile.llm.completed"
PROFILE_COMPLETED = "profile.completed"

# ----- Match planning -----

MATCH_REQUESTED = "match.requested"
MATCH_LLM_COMPLETED = "match.llm.completed"
MATCH_COMPLETED = "match.completed"

# ----- Resume composition -----

COMPOSE_REQUESTED = "compose.requested"
COMPOSE_LLM_COMPLETED = "compose.llm.completed"
COMPOSE_COMPLETED = "compose.completed"

# ----- Resume QA -----

QA_REQUESTED = "qa.requested"
QA_LLM_COMPLETED = "qa.llm.completed"
QA_COMPLETED = "qa.completed"

# ----- QA improver -----

QA_IMPROVE_REQUESTED = "qa_improve.requested"
QA_IMPROVE_LLM_COMPLETED = "qa_improve.llm.completed"
QA_IMPROVE_COMPLETED = "qa_improve.completed"

# ----- Cover letter -----

COVER_LETTER_REQUESTED = "cover_letter.requested"
COVER_LETTER_LLM_COMPLETED = "cover_letter.llm.completed"
COVER_LETTER_COMPLETED = "cover_letter.completed"
