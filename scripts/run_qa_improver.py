""" Run QA Improver Agent asynchronously using existing outputs."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import anyio

from agents.qa_improver_async import QAImproveAgent
from agents.resume_qa_async import ResumeQAResult
from core.llm_factory import get_async_llm_client
from core.models import JDAnalysisResult, ProfessionalProfile, TailoredResume
from core.obs import JsonRepoLogger, JsonStdoutLogger


def _load(path: Path, model_cls):
    try:
        return model_cls.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing required file: {path}") from exc


def main():
    p = ArgumentParser(
        description="Run QA and apply improvements to a tailored resume using existing outputs in ./out."
    )
    p.add_argument("--jd-json", type=Path, default=Path("out/jd.json"))
    p.add_argument("--profile-json", type=Path, default=Path("out/profile.json"))
    p.add_argument("--tailored-json", type=Path, default=Path("out/tailored.json"))
    p.add_argument(
        "--qa-json",
        type=Path,
        default=Path("out/qa.json"),
        help="Existing QA JSON; if missing, QA will be run",
    )
    p.add_argument(
        "--out-tailored",
        type=Path,
        default=Path("out/tailored_improved.json"),
        help="Where to write the improved TailoredResume JSON",
    )
    p.add_argument("--print", action="store_true", help="Print improved TailoredResume JSON")
    p.add_argument("--log-file", type=Path, help="Optional structured log file")
    args = p.parse_args()

    if args.log_file:
        logger = JsonStdoutLogger(service="scripts", env="dev", log_path=args.log_file)
    else:
        logger = JsonRepoLogger(service="scripts", env="dev", filename="qa_improver.log")

    jd = _load(args.jd_json, JDAnalysisResult)
    profile = _load(args.profile_json, ProfessionalProfile)
    tailored = _load(args.tailored_json, TailoredResume)
    qa_result = _load(args.qa_json, ResumeQAResult)
    llm = get_async_llm_client(logger=logger)

    async def _run():
        improver = QAImproveAgent(llm=llm, logger=logger)
        improved = await improver.improve(jd, profile, tailored, qa_result)
        return improved

    improved_resume = anyio.run(_run)

    if args.print or not args.out_tailored:
        print(improved_resume.model_dump_json(indent=2))

    if args.out_tailored:
        args.out_tailored.parent.mkdir(parents=True, exist_ok=True)
        args.out_tailored.write_text(improved_resume.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
