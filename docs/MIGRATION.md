# Migration Plan

## Kept (infrastructure useful for future reuse)
- Repository tooling (`Makefile`, `pyproject.toml`, `requirements.txt`).
- Base repo layout for tests and docs.

## Replaced
- Resume-tailoring agents and pipeline are deprecated in favor of GovGuard modules
  under `src/govguard/`.

## Added
- New GovGuard package with contracts, orchestrator, agents, gatekeeper, observability,
  demo scenarios, and registry fixtures.
- Docs: architecture, contracts, runbook, and this migration plan.
- Docker Compose stack with NATS event bus and GovGuard service.
- New unit/integration tests covering gating and rollback.

## Removed
- Legacy resume pipeline tests (`tests/test_*resume*`, `tests/test_jd_*`, etc.).
