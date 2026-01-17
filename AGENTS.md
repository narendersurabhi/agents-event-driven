# Agent Change Log

This file tracks changes made by automated agents. Update it with each set of
changes to provide context for future work.

## 2025-09-02
- Added centralized settings to load app configuration and CORS allowlists.
- Enhanced structured logging with context binding and redaction hooks.
- Updated lint configuration to align with Ruff's latest settings layout.
- Hardened LLM clients and workers with stricter typing, retry handling, and payload validation.
- Added package markers and test fixes to keep mypy/pytest green across scripts and UI.
- Refined pipeline runtime initialization and logging timestamps for compatibility.

## 2026-01-17
- Added candidate profile context loading and prompt injection to align resume agents.
- Updated QA, cover letter, match planning, and composer prompts to use candidate context.
- Documented candidate profile configuration options in the README.

## 2026-01-17
- Added GovGuard domain modules (contracts, agents, gatekeeper, orchestrator, demos, observability).
- Introduced deterministic fixtures, scenario runner, and rollback simulation.
- Added docs, docker-compose stack, and Makefile targets for demos and local deploy.
- Replaced legacy tests with governance unit/integration coverage.
