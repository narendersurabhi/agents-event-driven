"""Policy engine for gate decisions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from govguard.contracts.models import (
    CostLatencyMetrics,
    DriftBiasMetrics,
    EvalMetrics,
    GateDecision,
    ReleaseCandidate,
    SecurityFindings,
)
from govguard.contracts.types import GateDecisionType


@dataclass(slots=True)
class PolicyConfig:
    """Loaded policy configuration."""

    required_checks: dict[str, dict[str, list[str]]]
    thresholds: dict[str, Any]
    decision_tolerance: dict[str, str]

    @classmethod
    def load(cls, path: Path) -> PolicyConfig:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            required_checks=data["required_checks"],
            thresholds=data["thresholds"],
            decision_tolerance=data["decision_tolerance"],
        )


class Gatekeeper:
    """Policy-driven gatekeeper decision engine."""

    def __init__(self, policy: PolicyConfig) -> None:
        self._policy = policy

    def decide(self, candidate: ReleaseCandidate, results: dict[str, Any]) -> GateDecision:
        env = candidate.env.value
        tier_key = f"risk_tier_{candidate.risk_tier}"
        required = set(self._policy.required_checks[env][tier_key])
        reasons: list[str] = []
        warnings: list[str] = []
        completed_checks = set(results.keys())
        if candidate.lineage is not None:
            completed_checks.add("lineage")

        required_present = required.issubset(completed_checks)
        if not required_present:
            missing = required.difference(completed_checks)
            reasons.append(f"Missing required checks: {sorted(missing)}")

        required_checks_passed = required_present

        eval_metrics = results.get("eval")
        if isinstance(eval_metrics, EvalMetrics):
            thresholds = self._policy.thresholds["eval"]
            auroc_min = thresholds["auroc_min"][env]
            f1_min = thresholds["f1_min"][env]
            if eval_metrics.auroc is not None and eval_metrics.auroc < auroc_min:
                required_checks_passed = False
                reasons.append(f"AUROC {eval_metrics.auroc:.2f} below {auroc_min:.2f} threshold")
            if eval_metrics.f1_score is not None and eval_metrics.f1_score < f1_min:
                required_checks_passed = False
                reasons.append(f"F1 {eval_metrics.f1_score:.2f} below {f1_min:.2f} threshold")
            if eval_metrics.hallucination_flag:
                required_checks_passed = False
                reasons.append("Hallucination flag detected in LLM evaluation")
            if eval_metrics.groundedness is not None:
                ground_min = self._policy.thresholds["llm"]["groundedness_min"][env]
                if eval_metrics.groundedness < ground_min:
                    warnings.append(
                        f"Groundedness {eval_metrics.groundedness:.2f} below {ground_min:.2f}"
                    )
            if eval_metrics.citation_coverage is not None:
                cite_min = self._policy.thresholds["llm"]["citation_coverage_min"][env]
                if eval_metrics.citation_coverage < cite_min:
                    warnings.append(
                        f"Citation coverage {eval_metrics.citation_coverage:.2f} below {cite_min:.2f}"
                    )

        drift_metrics = results.get("drift_bias")
        if isinstance(drift_metrics, DriftBiasMetrics):
            psi_max = self._policy.thresholds["drift_bias"]["psi_max"][env]
            if drift_metrics.psi > psi_max:
                required_checks_passed = False
                reasons.append(f"Drift PSI {drift_metrics.psi:.2f} exceeds {psi_max:.2f}")

        security_metrics = results.get("security")
        if isinstance(security_metrics, SecurityFindings):
            if security_metrics.pii_leak_detected or security_metrics.prompt_injection_detected:
                required_checks_passed = False
                reasons.append("Security findings detected")

        cost_latency = results.get("cost_latency")
        if isinstance(cost_latency, CostLatencyMetrics):
            thresholds = self._policy.thresholds["cost_latency"]
            p95_max = thresholds["p95_ms_max"][env]
            cost_max = thresholds["cost_usd_per_1k_max"][env]
            if cost_latency.p95_ms > p95_max:
                required_checks_passed = False
                reasons.append(f"p95 latency {cost_latency.p95_ms:.0f}ms exceeds {p95_max:.0f}ms")
            if cost_latency.cost_usd_per_1k > cost_max:
                warnings.append(
                    f"Cost per 1k {cost_latency.cost_usd_per_1k:.3f} exceeds {cost_max:.3f}"
                )

        tolerance = self._policy.decision_tolerance[env]
        if required_checks_passed:
            decision = GateDecisionType.APPROVE
            if warnings and tolerance == "warnings":
                decision = GateDecisionType.APPROVE_WITH_WARNINGS
        else:
            decision = GateDecisionType.BLOCK

        if not reasons:
            reasons.append("All required checks satisfied")

        return GateDecision(
            candidate_id=candidate.candidate_id,
            decision=decision,
            reasons=reasons,
            required_checks_passed=required_checks_passed,
            warnings=warnings,
            metadata={"env": env, "risk_tier": candidate.risk_tier},
        )
