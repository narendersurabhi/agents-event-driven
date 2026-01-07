

""" Orchestrator script to run the full resume pipeline asynchronously. """

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import anyio

from scripts.linear_orchestrator import ResumePipelineOrchestrator
from core.llm_factory import get_async_llm_client, get_sync_llm_client
from core.obs import JsonRepoLogger, JsonStdoutLogger


def main():
    p = ArgumentParser(description="Run full resume pipeline (JD -> profile -> plan -> compose -> QA -> improve)")
    p.add_argument("--jd", type=Path, default=Path("jd.txt"), help="Path to raw JD text")
    p.add_argument("--resume", type=Path, default=Path("resume.txt"), help="Path to raw resume text")
    p.add_argument("--no-qa", action="store_true", help="Skip QA stage")
    p.add_argument("--no-improve", action="store_true", help="Skip improver stage")
    p.add_argument("--log-file", type=Path, help="Optional structured log file path")
    p.add_argument("--print", action="store_true", help="Print improved TailoredResume JSON")
    p.add_argument("--out", type=Path, default=Path("out/tailored_improved.json"), help="Where to write improved TailoredResume JSON")
    args = p.parse_args()

    if args.log_file:
        logger = JsonStdoutLogger(service="scripts", env="dev", log_path=args.log_file)
    else:
        logger = JsonRepoLogger(service="scripts", env="dev", filename="orchestrator.log")

    jd_text = args.jd.read_text(encoding="utf-8")
    resume_text = args.resume.read_text(encoding="utf-8")

    # Choose provider via factory; configured by LLM_PROVIDER env or override in factory call
    async_client = get_async_llm_client(logger=logger)
    sync_client = get_sync_llm_client(logger=logger)

    orch = ResumePipelineOrchestrator(
        async_llm=async_client,
        sync_llm=sync_client,
        logger=logger,
        run_qa=not args.no_qa,
        run_improver=not args.no_improve,
    )

    result = anyio.run(lambda: orch.run(jd_text, resume_text))

    if args.print or not args.out:
        print((result.improved or result.tailored).model_dump_json(indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text((result.improved or result.tailored).model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
