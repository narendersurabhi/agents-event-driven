""" Asynchronous QA-based Resume Improvement Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
import json

from pydantic import ValidationError

from agents.resume_qa_async import ResumeQAResult
from core.config import get_default_model
from core.llm_client import AsyncLLMClient
from core.models import JDAnalysisResult, ProfessionalProfile, TailoredResume
from core.obs import Logger, NullLogger

_SYSTEM = """
You improve a TailoredResume by applying QA suggestions and maximizing alignment with the target job.

You MUST return valid JSON matching the TailoredResume schema exactly:
- Same top-level shape as the input TailoredResume.
- No extra top-level keys.
- No markdown, no comments, no prose.

Assumptions and goals:
- The PROFILE is not exhaustive; the candidate has more experience than what is written.
- Assume the candidate reasonably has experience that matches the JOB ANALYSIS unless it clearly conflicts with the PROFILE.
- Your job is to surface and articulate that experience in the TailoredResume.

Inference rules:
- You MAY infer missing but plausible details that are strongly suggested by:
  - The candidate’s seniority, domains, and tech stack in PROFILE.
  - The requirements and context in JOB ANALYSIS.
  - The issues and gaps called out in QA SUGGESTIONS.
- Plausible inferences include:
  - Concrete tools/frameworks that are natural for this stack (e.g., PyTorch for deep learning, Airflow for data pipelines, SHAP for explainability).
  - Typical responsibilities at this level (mentoring, architecture reviews, roadmap planning, cross-team leadership).
  - Metrics and impact (latency, throughput, p95, cost savings, accuracy/recall improvements, reviewer productivity) consistent with the described scale.
- Prefer strengthening and enriching existing roles over inventing entirely new employers, job titles, or time periods.

Constraints:
- Do NOT contradict explicit facts in PROFILE or CURRENT TAILORED RESUME (titles, companies, dates, domains).
- When a JD requirement is clearly outside the candidate’s history (e.g., a brand-new domain with no overlap), prefer framing as:
  - “familiarity with…”, “supporting…”, “collaborated with…”, or “applied similar techniques to…”
  rather than claiming many years of deep ownership.
- Keep everything coherent with the candidate’s overall narrative, domain focus, and seniority.

Output:
- Return ONLY the updated TailoredResume JSON.
- The JSON MUST be syntactically valid and match the TailoredResume schema.
"""

# _USER = """
# JOB ANALYSIS:
# {jd_json}

# PROFILE (truth source):
# {profile_json}

# CURRENT TAILORED RESUME:
# {resume_json}

# QA SUGGESTIONS (address these):
# {suggestions_json}

# Update the TailoredResume to incorporate the suggestions while staying truthful to the profile.
# Return only TailoredResume JSON.
# """

_USER = """
JOB ANALYSIS (target role and requirements):
{jd_json}

PROFILE (anchor for domains, seniority, and constraints – this is NOT a full history of the candidate):
{profile_json}

CURRENT TAILORED RESUME (existing version to improve):
{resume_json}

QA SUGGESTIONS (gaps and improvements to address):
{suggestions_json}

Task:
Update the TailoredResume to:
- Fully address the QA SUGGESTIONS.
- Maximize coverage of JOB ANALYSIS must-have and nice-to-have skills.
- Assume the candidate has the necessary experience that reasonably fits their PROFILE and the JOB ANALYSIS, even if it is not explicitly described.
- Infer and surface plausible missing details that a candidate with this PROFILE would likely have, such as:
  - Specific ML techniques and models (e.g., logistic regression, gradient boosting, deep learning frameworks, GNNs if compatible with their work).
  - Platform and MLOps capabilities (e.g., feature stores, CI/CD, canary/shadow deployments, monitoring, SLOs, drift/bias detection).
  - Responsible AI and interpretability practices (e.g., SHAP, LIME, Clarify, fairness and bias controls).
  - Leadership and collaboration (mentoring, code/design reviews, cross-functional roadmapping, stakeholder communication).

Guidance on how to phrase inferred experience:
- If the JD requirement is strongly aligned with the PROFILE (e.g., ML platforms, large-scale fraud detection, healthcare analytics), you may describe it as concrete, owned experience in the corresponding roles.
- If the requirement is adjacent but not directly evidenced (e.g., a specific framework or tool in a familiar area), you may treat it as used within those roles.
- If the requirement is in a new domain (e.g., crypto/blockchain when PROFILE is healthcare/fraud), prefer phrasing such as:
  - “applied similar risk modeling techniques to [candidate’s domain]”
  - “leveraged methods that transfer well to [job domain]”
  - or show strong interest and self-driven learning, unless PROFILE clearly supports production ownership.

Structural requirements:
- Preserve the TailoredResume schema: keep the same top-level keys and overall JSON structure as CURRENT TAILORED RESUME.
- You may add, remove, or rewrite bullet points and skills within that structure to best align with the JOB ANALYSIS and QA SUGGESTIONS.
- Do NOT include explanations, comments, or any text outside the JSON.

Return ONLY the updated TailoredResume JSON.
"""


class QAImproveError(RuntimeError):
    """Base error for QA improvement failures."""


class QAImproveInvalid(QAImproveError):
    """Raised when the improved resume payload fails validation or parsing."""


@dataclass(slots=True)
class QAImproveAgent:
    """Agent that applies QA suggestions to produce an improved TailoredResume."""

    llm: AsyncLLMClient
    model: str = field(default_factory=get_default_model)
    logger: Logger = field(default_factory=NullLogger)

    async def improve(
        self,
        jd: JDAnalysisResult,
        profile: ProfessionalProfile,
        resume: TailoredResume,
        qa: ResumeQAResult,
    ) -> TailoredResume:
        """Return an improved TailoredResume that addresses QA suggestions."""
        msgs = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _USER.format(
                    jd_json=jd.model_dump_json(),
                    profile_json=profile.model_dump_json(),
                    resume_json=resume.model_dump_json(),
                    suggestions_json=json.dumps(qa.suggestions, ensure_ascii=False, indent=2),
                ),
            },
        ]

        raw = await self.llm.chat(messages=msgs, model=self.model, temperature=0.1)
        self.logger.info("qa_improve.raw_response", raw=raw, raw_chars=len(raw))
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end == -1:
                raise QAImproveInvalid("No JSON detected in improver output") from exc
            data = json.loads(raw[start : end + 1])

        try:
            return TailoredResume.model_validate(data)
        except ValidationError as e:
            raise QAImproveInvalid(f"Improved resume validation failed: {e}") from e
