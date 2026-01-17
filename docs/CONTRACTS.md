# GovGuard Event Contracts

All events are defined in `src/govguard/contracts/events.py` and use Pydantic models.
Each event payload is wrapped in the internal event bus envelope.

## Event Types

- `release.candidate.created`
- `eval.started`
- `eval.completed`
- `eval.failed`
- `eval.warning`
- `gate.decision.made`
- `deploy.started`
- `deploy.approved`
- `deploy.blocked`
- `deploy.completed`
- `monitoring.regression.detected`
- `deploy.rolled_back`

## Example Payloads

### release.candidate.created

```json
{
  "candidate": {
    "candidate_id": "c4c2b2e0-2b29-4f3c-95b8-3e6a519c8f90",
    "artifact_refs": [
      {"type": "MODEL", "version": "1.3.0", "digest": "sha256:abc"}
    ],
    "baseline_ref": {"type": "MODEL", "version": "1.2.5", "digest": "sha256:base"},
    "env": "prod",
    "risk_tier": 1,
    "lineage": {
      "code_sha": "deadbeef",
      "data_snapshot_id": "snapshot-2025-01-17",
      "feature_schema_version": "v7",
      "build_timestamp": "2025-01-17T10:00:00Z"
    }
  }
}
```

### eval.completed

```json
{
  "candidate_id": "c4c2b2e0-2b29-4f3c-95b8-3e6a519c8f90",
  "check_name": "security",
  "metrics": {
    "pii_leak_detected": false,
    "prompt_injection_detected": false,
    "findings": []
  }
}
```

### gate.decision.made

```json
{
  "decision": {
    "candidate_id": "c4c2b2e0-2b29-4f3c-95b8-3e6a519c8f90",
    "decision": "APPROVE",
    "reasons": ["All required checks satisfied"],
    "required_checks_passed": true,
    "warnings": [],
    "metadata": {"env": "prod", "risk_tier": 1}
  }
}
```
