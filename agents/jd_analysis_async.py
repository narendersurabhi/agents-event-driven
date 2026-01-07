""" Asynchronous Job Description Analysis Agent."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from core.llm_client import AsyncLLMClient
from core.models import JDAnalysisResult
from core.config import get_default_model

from .jd_analysis import (  # reuse prompts and errors
    AGENT_SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    JDAnalysisError,
    JDAnalysisInvalidResponse,
)


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AsyncJDAnalysisAgent:
    """Async agent for analyzing job descriptions.

    Same contract as JDAnalysisAgent, but fully async and event-loop friendly.
    """

    llm: AsyncLLMClient
    model: str = field(default_factory=get_default_model)

    def _build_messages(self, job_description: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(job_description=job_description.strip()),
            },
        ]

    def _parse_json(self, raw: str) -> dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1:
                logger.error("AsyncJDAnalysisAgent: no JSON object detected in output")
                raise JDAnalysisInvalidResponse("No JSON detected in model output") from e

            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError as e1:
                logger.exception("AsyncJDAnalysisAgent: failed to parse JSON substring")
                raise JDAnalysisInvalidResponse("Malformed JSON in model output") from e1

    async def analyze(self, job_description: str) -> JDAnalysisResult:
        """ Analyze a job description asynchronously."""
        jd = job_description.strip()
        if not jd:
            raise JDAnalysisError("job_description must be a non-empty string")

        messages = self._build_messages(jd)

        logger.info("AsyncJDAnalysisAgent: analyzing JD (%d chars)", len(jd))

        raw = await self.llm.chat(
            messages=messages,
            model=self.model,
            temperature=0.1,
        )
        logger.debug("AsyncJDAnalysisAgent: raw LLM output: %s", raw[:500])

        data = self._parse_json(raw)

        try:
            result = JDAnalysisResult.model_validate(data)
        except ValidationError as e:
            logger.exception("AsyncJDAnalysisAgent: validation failed")
            raise JDAnalysisInvalidResponse(f"Validation failed: {e}") from e

        logger.info(
            "AsyncJDAnalysisAgent: success role_title=%s company=%s",
            result.role_title,
            result.company,
        )
        return result
