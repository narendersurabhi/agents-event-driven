from agents.jd_analysis import (
    JDAnalysisAgent,
    JDAnalysisInvalidResponse,
    JDAnalysisError,
)
from core.models import JDAnalysisResult


class FakeLLMGood:
    def chat(self, messages: list[dict[str, str]], model: str, temperature: float = 0.0) -> str:
        return """
        {
          "role_title": "Senior Machine Learning Engineer",
          "company": "TestCorp",
          "seniority_level": "Senior",
          "must_have_skills": ["Python", "ML", "AWS"],
          "nice_to_have_skills": ["Spark"],
          "notes_for_resume": "- Highlight large-scale ML pipelines\\n- Show impact metrics"
        }
        """


class FakeLLMBadJSON:
    def chat(self, messages: list[dict[str, str]], model: str, temperature: float = 0.0) -> str:
        return "I am not JSON at all."


def test_analyze_happy_path() -> None:
    agent = JDAnalysisAgent(llm=FakeLLMGood())
    result = agent.analyze("some JD text")
    assert isinstance(result, JDAnalysisResult)
    assert result.role_title == "Senior Machine Learning Engineer"
    assert "Python" in result.must_have_skills
    assert result.company == "TestCorp"


def test_analyze_invalid_json_raises() -> None:
    agent = JDAnalysisAgent(llm=FakeLLMBadJSON())
    try:
        agent.analyze("some JD text")
    except JDAnalysisInvalidResponse:
        return
    assert False, "Expected JDAnalysisInvalidResponse"


def test_analyze_empty_input_raises() -> None:
    agent = JDAnalysisAgent(llm=FakeLLMGood())
    try:
        agent.analyze("   ")
    except JDAnalysisError:
        return
    assert False, "Expected JDAnalysisError on empty input"
