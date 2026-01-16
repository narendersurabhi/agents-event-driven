from argparse import ArgumentParser, FileType
import logging

from agents.jd_analysis import JDAnalysisAgent
from core.config import get_default_model
from core.llm_client import OpenAILLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = ArgumentParser(description="Demo CLI for JDAnalysisAgent (sync).")
    parser.add_argument(
        "file",
        type=FileType("r"),
        help="Path to a text file containing the job description.",
    )
    args = parser.parse_args()

    jd_text = args.file.read()

    llm = OpenAILLMClient()
    agent = JDAnalysisAgent(llm=llm, model=get_default_model())

    logger.info("Analyzing job description...")
    result = agent.analyze(jd_text)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
