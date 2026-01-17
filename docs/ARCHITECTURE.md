# GovGuard Architecture

## Components

- **Orchestrator**: Drives the Release Candidate lifecycle and publishes next-step events.
- **Agents**: Deterministic checks for evaluation, drift/bias, security, cost/latency.
- **Gatekeeper**: Policy-driven decision engine returning APPROVE/BLOCK/APPROVE_WITH_WARNINGS.
- **Rollback Agent**: Consumes monitoring regression signals to trigger rollback.
- **Event Bus**: In-memory for local/test; pluggable for external buses.
- **Observability**: Structured JSON logs, OpenTelemetry tracing, Prometheus metrics.

## Event Flow (happy path)

```
release.candidate.created
        │
        ▼
    eval.started  ──> eval.completed (per agent)
        │
        ▼
 gate.decision.made
        │
        ├─ deploy.approved
        ├─ deploy.started
        └─ deploy.completed
```

## Event Flow (blocked)

```
release.candidate.created
        │
        ▼
    eval.completed (security flags)
        │
        ▼
 gate.decision.made (BLOCK)
        │
        └─ deploy.blocked
```

## Event Flow (rollback)

```
release.candidate.created
        │
        ▼
 gate.decision.made (APPROVE)
        │
        └─ deploy.completed
                │
                ▼
 monitoring.regression.detected
                │
                ▼
 deploy.rolled_back
```
