# GovGuard: Model Governance & Release Gatekeeper

GovGuard is an event-driven, agentic workflow for ML + LLM model governance. Every
training run or prompt/retrieval config change creates a Release Candidate, triggers
deterministic evaluation agents, and produces a policy-driven Gate Decision
(APPROVE / BLOCK / APPROVE_WITH_WARNINGS). Approved candidates can deploy (simulated),
and post-deploy monitoring can trigger rollbacks (simulated).

## Repo layout

- `src/govguard/contracts/` – event + domain models (Pydantic)
- `src/govguard/orchestrator/` – workflow orchestration + event bus
- `src/govguard/agents/` – deterministic evaluation, security, drift, cost/latency, rollback
- `src/govguard/gatekeeper/` – policy engine + thresholds
- `src/govguard/registry/` – fixture store (deterministic artifacts for demos/tests)
- `src/govguard/observability/` – logging, tracing, metrics
- `src/govguard/demo/` – scenario runners
- `docs/` – architecture + contracts + runbook
- `tests/` – unit + integration tests

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run demo scenarios

```bash
make demo-happy
make demo-blocked
make demo-rollback
```

## Testing

```bash
make lint
make typecheck
make test
```

## Local deploy (Docker Compose)

```bash
make compose-up
make compose-down
```

See `docs/RUNBOOK.md` for additional usage and scenario details.
