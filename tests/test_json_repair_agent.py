import importlib
from typing import Any

from agents.json_repair import JsonRepairAgent
from core.json_repair import JsonRepairAgent as CoreJsonRepairAgent


class FakeLLM:
    def __init__(self, response: str = "{}"):
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, model, temperature=0.0, **kwargs) -> str:  # noqa: ANN001
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "kwargs": kwargs,
            }
        )
        return self.response


def test_agents_json_repair_is_backwards_compatible_import() -> None:
    assert JsonRepairAgent is CoreJsonRepairAgent

    mod = importlib.import_module("agents.json_repair")
    assert getattr(mod, "__all__", None) == ["JsonRepairAgent"]


def test_json_repair_builds_expected_prompts_and_passes_req_id() -> None:
    llm = FakeLLM(response='{"ok": true}')
    agent = JsonRepairAgent(llm=llm, model="test-model")

    raw = "{bad json"
    schema = "class Foo(BaseModel):\n    a: int"
    out = agent.repair(raw=raw, schema_text=schema, error="boom", req_id="req-123")

    assert out == '{"ok": true}'
    assert len(llm.calls) == 1
    call = llm.calls[0]

    assert call["model"] == "test-model"
    assert call["temperature"] == 0.0
    assert call["kwargs"] == {"req_id": "req-123"}

    messages = call["messages"]
    assert messages[0]["role"] == "system"
    assert schema in messages[0]["content"]

    assert messages[1]["role"] == "user"
    assert "Original output:" in messages[1]["content"]
    assert raw in messages[1]["content"]
    assert "Parser/validation error:" in messages[1]["content"]
    assert "boom" in messages[1]["content"]


def test_json_repair_omits_error_and_req_id_when_not_provided() -> None:
    llm = FakeLLM(response="[]")
    agent = JsonRepairAgent(llm=llm, model="test-model")

    out = agent.repair(raw="{}", schema_text="{}", error=None, req_id=None)

    assert out == "[]"
    call = llm.calls[0]
    assert call["kwargs"] == {}
    assert "Parser/validation error:" not in call["messages"][1]["content"]
