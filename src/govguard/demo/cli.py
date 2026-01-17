"""CLI entrypoint for GovGuard demos."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from govguard.demo import fixtures
from govguard.demo.runner import run_scenario


def main() -> None:
    parser = ArgumentParser(description="Run GovGuard demo scenarios.")
    parser.add_argument(
        "scenario",
        choices=["happy", "blocked", "rollback"],
        help="Scenario to run.",
    )
    parser.add_argument(
        "--policy",
        default=Path("src/govguard/gatekeeper/policy.yaml"),
        type=Path,
        help="Path to policy configuration.",
    )
    args = parser.parse_args()

    if args.scenario == "happy":
        scenario = fixtures.happy_path()
    elif args.scenario == "blocked":
        scenario = fixtures.blocked_path()
    else:
        scenario = fixtures.rollback_path()

    result = run_scenario(scenario, args.policy)
    print(f"Scenario {args.scenario} completed for candidate {result.candidate_id}")


if __name__ == "__main__":
    main()
