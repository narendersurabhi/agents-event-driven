"""LLM-powered JSON repair utility.

This is infrastructure (not a domain agent): given malformed JSON and a target
schema description, it asks an LLM to emit corrected JSON-only output.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.llm_client import LLMClient


@dataclass(slots=True)
class JsonRepairAgent:
    """Repair malformed JSON given an expected schema description.

    The schema is provided as free-form text (e.g., Pydantic-style class
    definitions or JSON Schema). Callers are responsible for validating the
    repaired JSON against their own Pydantic models.
    """

    llm: LLMClient
    model: str

    def repair(
        self,
        raw: str,
        schema_text: str,
        error: str | None = None,
        req_id: str | None = None,
    ) -> str:
        """Return a best-effort repaired JSON string.

        The LLM is instructed to output JSON only, with no commentary or
        markdown wrapping. Callers should still run json.loads + validation
        and handle any remaining failures.
        """

        system = f"""
You are a strict JSON repair tool.
You receive invalid or partially valid JSON that was intended to match
this target schema (Pydantic-style description):

{schema_text}

Your job:
- Return a single valid JSON object or array that best matches the schema.
- Do NOT invent new fields beyond the schema unless absolutely necessary.
- If values are missing or unclear, use null or an empty list/string.
- Output JSON only, with no markdown, no backticks, and no commentary.
"""

        parts = [
            "The following text is the model's output that failed to parse or validate as JSON.",
            "Return a corrected JSON version.",
            "",
            "Original output:",
            raw,
        ]
        if error:
            parts.extend(["", "Parser/validation error:", error])
        user = "\n".join(parts)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        # Temperature 0 to keep the repair as deterministic as possible.
        extra = {"req_id": req_id} if req_id else {}
        return self.llm.chat(messages=messages, model=self.model, temperature=0.0, **extra)
