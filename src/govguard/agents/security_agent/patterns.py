"""Security detection helpers."""

from __future__ import annotations

import re

from govguard.contracts.models import SecurityFindings

_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_PATTERN = re.compile(r"\b\d{3}[-.]\d{3}[-.]\d{4}\b")
_PROMPT_INJECTION_PATTERN = re.compile(r"ignore previous|system prompt|exfiltrate", re.I)


def detect_security_findings(text: str) -> SecurityFindings:
    """Detect PII leakage and prompt injection indicators in text."""
    findings: list[str] = []
    pii_leak = False
    if _SSN_PATTERN.search(text) or _EMAIL_PATTERN.search(text) or _PHONE_PATTERN.search(text):
        pii_leak = True
        findings.append("Potential PII leakage detected")
    prompt_injection = False
    if _PROMPT_INJECTION_PATTERN.search(text):
        prompt_injection = True
        findings.append("Prompt injection pattern detected")
    return SecurityFindings(
        pii_leak_detected=pii_leak,
        prompt_injection_detected=prompt_injection,
        findings=findings,
    )
