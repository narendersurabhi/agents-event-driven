from __future__ import annotations

import json
from typing import Any


def parse_json_object(raw: str, error_cls: type[Exception]) -> dict[str, Any]:
    """Extract JSON object from a raw model string, raising error_cls on failure.

    Tries full-string JSON parse first, then extracts the first {...} block. On failure,
    raises error_cls with the original exception chained.
    """

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
        raise error_cls("Expected JSON object in model output")
    except json.JSONDecodeError as exc:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            raise error_cls("No JSON detected in model output") from exc
        try:
            data = json.loads(raw[start : end + 1])
            if isinstance(data, dict):
                return data
            raise error_cls("Expected JSON object in model output")
        except json.JSONDecodeError as exc2:
            raise error_cls("Malformed JSON in model output") from exc2
