# GovGuard Runbook

## Prerequisites

- Python 3.11+
- `pip install -r requirements.txt`

## Run demo scenarios

```bash
make demo-happy
make demo-blocked
make demo-rollback
```

The demos print a candidate ID on completion. Logs are JSON-formatted. Metrics are
exposed on `http://localhost:8005/metrics` while the demo is running.

## Interpret outputs

- `gate.decision.made` provides the policy decision and reasons.
- `deploy.blocked` indicates an explicit block.
- `deploy.rolled_back` indicates a simulated rollback due to regression.

## Add a new agent/check

1. Add a new agent in `src/govguard/agents/<agent_name>/`.
2. Register fixtures in `src/govguard/registry/fixture_store.py`.
3. Add the agent to the worker list in `src/govguard/demo/runner.py`.
4. Update policy thresholds in `src/govguard/gatekeeper/policy.yaml`.
5. Add unit + integration tests under `tests/`.
