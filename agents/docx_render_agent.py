"""Agent for rendering a DOCX document from structured data using docxtpl.

This is intentionally LLM-free: it takes a template (.docx as bytes) and a
Pydantic model or dict (e.g., TailoredResume) and returns rendered DOCX bytes.

Templates can reference root fields directly, e.g.:
    {{ full_name }} / {{ summary }} / {{ experience[0].title }} ...
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import io
from types import SimpleNamespace
from typing import Any

from docxtpl import DocxTemplate
from jinja2 import Environment
from pydantic import BaseModel


def _default_jinja_env() -> Environment:
    """Return a simple Jinja2 environment for docxtpl rendering."""
    # Enable autoescape so XML special chars like '&', '<', '>' are
    # properly escaped (e.g., '&' -> '&amp;') in the underlying document.xml.
    # This ensures values such as "LLMs & GenAI" render correctly in Word.
    env = Environment(autoescape=True)
    return env


def _coerce_to_base(data: Any) -> Any:
    if isinstance(data, BaseModel):
        return data.model_dump()
    if isinstance(data, Mapping):
        return dict(data)
    return {"value": data}


def _normalize_skills(skills: Any) -> Any:
    # Expect a list of objects with .name and .items.
    if not isinstance(skills, list):
        return skills
    normalized: list[Any] = []
    for skill in skills:
        if isinstance(skill, Mapping):
            name = skill.get("name") or ""
            items = skill.get("items") or []
        else:
            name = str(skill)
            items = []

        if not isinstance(items, list):
            items = list(items) if hasattr(items, "__iter__") else [items]

        # Use a simple object so Jinja `cat.items` resolves to the list,
        # not dict.items method.
        normalized.append(SimpleNamespace(name=name, items=items))
    return normalized


def _normalize_education(education: Any) -> Any:
    # Allow list[str] or list[dict].
    if not isinstance(education, list) or not education:
        return education
    if isinstance(education[0], Mapping):
        return education
    return [
        {
            "institution": str(entry),
            "degree": "",
            "start_date": None,
            "end_date": None,
            "location": None,
        }
        for entry in education
    ]


def _normalize_certifications(certifications: Any) -> Any:
    # Allow list[str] or list[dict].
    if not isinstance(certifications, list) or not certifications:
        return certifications
    if isinstance(certifications[0], Mapping):
        return certifications
    return [{"name": str(c), "issuer": None, "year": None} for c in certifications]


def _normalize_docx_payload(base: Any) -> Any:
    """Normalize payload shape so templates are resilient to older job snapshots."""
    if not isinstance(base, dict):
        return base
    base["skills"] = _normalize_skills(base.get("skills"))
    base["education"] = _normalize_education(base.get("education"))
    base["certifications"] = _normalize_certifications(base.get("certifications"))
    return base


def _build_context(base: Any) -> dict[str, Any]:
    # Context exposes root keys only (no 'data' prefix).
    if isinstance(base, dict):
        return dict(base)
    return {"value": base}


@dataclass(slots=True)
class DocxRenderAgent:
    """Render a DOCX file from a template and structured data."""

    jinja_env: Environment = field(default_factory=_default_jinja_env)

    def render(self, template_bytes: bytes, data: Any) -> bytes:
        """Render DOCX bytes given a template and data.

        - template_bytes: raw .docx template bytes
        - data: a Pydantic BaseModel, dict, or other JSON-serializable object
        """
        tpl_stream = io.BytesIO(template_bytes)
        tpl = DocxTemplate(tpl_stream)

        base = _normalize_docx_payload(_coerce_to_base(data))
        ctx = _build_context(base)

        tpl.render(ctx, jinja_env=self.jinja_env)
        out_stream = io.BytesIO()
        tpl.save(out_stream)
        return out_stream.getvalue()
