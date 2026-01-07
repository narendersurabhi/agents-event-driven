from argparse import ArgumentParser, FileType
from pathlib import Path
import json
import uuid

from agents.jd_analysis import JDAnalysisAgent
from agents.match_planner import MatchPlannerAgent
from agents.resume_composer import ResumeComposerAgent
from agents.profile_from_resume import ProfileFromResumeAgent
# from agents.resume_qa import ResumeQAAgent  # if you wired sync QA
from core.llm_client import OpenAILLMClient
from agents.resume_qa_async import AsyncResumeQAAgent  # make sure the file exists
from core.llm_factory import get_async_llm_client
import anyio

from core.obs import JsonStdoutLogger, JsonRepoLogger

def main():
    p = ArgumentParser(description="Tailor resume from raw JD text + raw resume text.")
    p.add_argument("--jd", type=FileType("r"), required=True, help="Path to JD text file")
    p.add_argument("--resume", type=FileType("r"), required=True, help="Path to resume text file")
    p.add_argument("--print-text", action="store_true", help="Also print flattened resume_text")
    p.add_argument("--out-txt", type=Path, help="Path to save flattened resume_text (e.g., out/resume.txt)")
    p.add_argument("--out-json", type=Path, help="Path to save full TailoredResume JSON (e.g., out/resume.json)")
    p.add_argument("--run-qa", action="store_true", help="Run QA and print/save results")
    p.add_argument("--out-qa", type=Path, help="Path to save QA JSON (e.g., out/qa.json)")
    p.add_argument("--log-file", type=Path, help="Path to append structured logs (JSON lines)")
    args = p.parse_args()

    jd_text = args.jd.read()
    resume_text = args.resume.read()

    if args.log_file:
        logger = JsonStdoutLogger(service="scripts", env="dev", log_path=args.log_file)
    else:
        logger = JsonRepoLogger(service="scripts", env="dev", filename="tailor_from_text.log")
    llm = OpenAILLMClient(logger=logger)

    # JD → analysis
    jd_agent = JDAnalysisAgent(llm=llm)
    jd = jd_agent.analyze(jd_text)

    # after you compute jd and profile:
    out_dir = Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "jd.json").write_text(jd.model_dump_json(indent=2), encoding="utf-8")


    # Resume text → canonical profile
    profile_agent = ProfileFromResumeAgent(llm=llm)
    profile = profile_agent.extract(resume_text)
    (out_dir / "profile.json").write_text(profile.model_dump_json(indent=2), encoding="utf-8")

    # Plan → Compose
    planner = MatchPlannerAgent(llm=llm)
    plan = planner.plan(jd, profile)
    (out_dir / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")


    composer = ResumeComposerAgent(llm=llm)
    tailored = composer.compose(jd, profile, plan)
    (out_dir / "tailored.json").write_text(tailored.model_dump_json(indent=2), encoding="utf-8")

    qa_result = None
    if args.run_qa:
        async def _run_qa():
            allm = get_async_llm_client(logger=logger)
            qa = AsyncResumeQAAgent(llm=allm, logger=logger)
            return await qa.review(jd, profile, tailored)
        qa_result = anyio.run(_run_qa)

    # Print to console (optional)
    if args.print_text:
        print("=== resume_text ===")
        print(tailored.resume_text)
        print("\n=== QA ===")
        print(qa_result.model_dump_json(indent=2))

    # Save outputs (optional)
    if args.out_txt:
        args.out_txt.parent.mkdir(parents=True, exist_ok=True)
        args.out_txt.write_text(tailored.resume_text, encoding="utf-8")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(tailored.model_dump_json(indent=2), encoding="utf-8")

    if args.out_qa:
        args.out_qa.parent.mkdir(parents=True, exist_ok=True)
        args.out_qa.write_text(qa_result.model_dump_json(indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
