from typing import Any

import pytest

from rag.agent import Agent, MAX_TOOL_ROUNDS


class RepeatingToolChat:
    def __init__(self) -> None:
        self.calls = 0

    async def generate_stream(self, messages, model: str, **kwargs: Any):
        self.calls += 1
        if kwargs.get("tools"):
            yield {
                "content": None,
                "tools": [
                    {
                        "id": f"call-{self.calls}",
                        "type": "function",
                        "function": {
                            "name": "semantic_search",
                            "arguments": '{"query": "summarize the document"}',
                        },
                    }
                ],
            }
            return

        assert kwargs.get("tool_choice") == "none"
        yield {"content": "Final grounded answer.", "tools": None}


@pytest.mark.asyncio
async def test_agent_generates_final_answer_after_retrieval_limit():
    chat = RepeatingToolChat()
    agent = Agent(model="test", _chat=chat, _search=object(), _embed=object())

    async def fake_execute(tool_name: str, **kwargs: Any) -> str:
        return "Retrieved document evidence."

    agent.execute = fake_execute  # type: ignore[method-assign]
    messages = [{"role": "user", "content": "Summarize the document."}]

    output = []
    async for chunk in agent.generate(messages):
        output.append(chunk)

    assert chat.calls == MAX_TOOL_ROUNDS + 1
    assert "".join(output) == "Final grounded answer."
    assert messages[-1]["role"] == "system"
