from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import anyio

from agents.resume_qa_async import AsyncResumeQAAgent
from core.llm_factory import get_async_llm_client
from core.models import JDAnalysisResult, ProfessionalProfile, TailoredResume
from core.obs import JsonRepoLogger, JsonStdoutLogger


def _load_model(path: Path, model_cls):
    try:
        return model_cls.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing required file: {path}") from exc


def main():
    p = ArgumentParser(
        description="Run resume QA against existing JD/profile/tailored JSON outputs."
    )
    p.add_argument(
        "--jd-json",
        type=Path,
        required=True,
        help="Path to JDAnalysisResult JSON (e.g., out/jd.json)",
    )
    p.add_argument(
        "--profile-json",
        type=Path,
        required=True,
        help="Path to ProfessionalProfile JSON (e.g., out/profile.json)",
    )
    p.add_argument(
        "--tailored-json",
        type=Path,
        required=True,
        help="Path to TailoredResume JSON (e.g., out/tailored.json)",
    )
    p.add_argument("--out-qa", type=Path, help="Path to save QA JSON output (e.g., out/qa.json)")
    p.add_argument("--print", action="store_true", help="Print the QA JSON to stdout")
    p.add_argument(
        "--log-file",
        type=Path,
        help="Optional path for structured logs; defaults to repo logs directory",
    )
    args = p.parse_args()

    if args.log_file:
        logger = JsonStdoutLogger(service="scripts", env="dev", log_path=args.log_file)
    else:
        logger = JsonRepoLogger(service="scripts", env="dev", filename="resume_qa.log")

    llm = get_async_llm_client(logger=logger)
    qa_agent = AsyncResumeQAAgent(llm=llm, logger=logger)

    jd = _load_model(args.jd_json, JDAnalysisResult)
    profile = _load_model(args.profile_json, ProfessionalProfile)
    tailored = _load_model(args.tailored_json, TailoredResume)

    async def _run():
        return await qa_agent.review(jd, profile, tailored)

    qa_result = anyio.run(_run)

    if args.print or not args.out_qa:
        print(qa_result.model_dump_json(indent=2))

    if args.out_qa:
        args.out_qa.parent.mkdir(parents=True, exist_ok=True)
        args.out_qa.write_text(qa_result.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
