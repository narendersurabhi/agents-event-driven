import pytest

from agents.jd_analysis import (
    JDAnalysisError,
    JDAnalysisInvalidResponse,
)
from agents.jd_analysis_async import AsyncJDAnalysisAgent
from core.models import JDAnalysisResult


class FakeAsyncLLMGood:
    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        **kwargs: object,
    ) -> str:
        return """
        {
          "role_title": "Staff Data Scientist",
          "company": "AsyncCorp",
          "seniority_level": "Staff",
          "must_have_skills": ["Python", "Statistics"],
          "nice_to_have_skills": [],
          "notes_for_resume": "- Emphasize experimentation pipelines"
        }
        """


class FakeAsyncLLMBadJSON:
    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.0,
        **kwargs: object,
    ) -> str:
        return "Definitely not JSON."


@pytest.mark.asyncio
async def test_async_analyze_happy_path() -> None:
    agent = AsyncJDAnalysisAgent(llm=FakeAsyncLLMGood(), model="test-model")
    result = await agent.analyze("some JD text")
    assert isinstance(result, JDAnalysisResult)
    assert result.role_title == "Staff Data Scientist"
    assert result.company == "AsyncCorp"


@pytest.mark.asyncio
async def test_async_analyze_invalid_json_raises() -> None:
    agent = AsyncJDAnalysisAgent(llm=FakeAsyncLLMBadJSON(), model="test-model")
    with pytest.raises(JDAnalysisInvalidResponse):
        await agent.analyze("some JD text")


@pytest.mark.asyncio
async def test_async_analyze_empty_input_raises() -> None:
    agent = AsyncJDAnalysisAgent(llm=FakeAsyncLLMGood(), model="test-model")
    with pytest.raises(JDAnalysisError):
        await agent.analyze("   ")
