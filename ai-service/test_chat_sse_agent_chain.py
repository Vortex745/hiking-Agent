import asyncio
import json

from api import chat
from api.models import ChatRequest, CurrentLocationPayload


class FakeMemory:
    def __init__(self) -> None:
        self.messages = [{"role": "assistant", "content": "已有上下文"}]

    def get_messages(self) -> list[dict[str, str]]:
        return list(self.messages)

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})


class FakeAgent:
    invocation: dict[str, object] | None = None

    def __init__(self, *, llm_config: object) -> None:
        self.llm_config = llm_config

    async def aexecute_stream(
        self, message: str, history: list[dict[str, str]], **kwargs: object
    ):
        type(self).invocation = {
            "message": message,
            "history": history,
            "kwargs": kwargs,
        }
        yield {"type": "thought", "content": "分析需求"}
        yield {"type": "tool_call", "content": "route_search"}
        yield {"type": "tool_result", "content": "找到候选路线"}
        yield {"type": "text", "content": "推荐西湖群山路线。"}
        yield {"type": "done", "content": ""}


async def _stream_body(response: object) -> str:
    chunks: list[bytes] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.encode() if isinstance(chunk, str) else chunk)
    return b"".join(chunks).decode()


def test_chat_sse_runs_agent_chain_and_persists_reply(monkeypatch) -> None:
    memory = FakeMemory()
    FakeAgent.invocation = None
    monkeypatch.setattr(chat, "_get_memory", lambda _chat_id: memory)
    monkeypatch.setattr(chat, "_has_llm_api_key", lambda _req: True)
    monkeypatch.setattr(chat, "_get_agent_class", lambda: FakeAgent)
    monkeypatch.setattr(chat, "get_store", lambda: object())

    request = ChatRequest(
        message="帮我规划本周末杭州徒步",
        chat_id="agent-chain-test",
        scenario="route_plan",
        current_location=CurrentLocationPayload(city="杭州", source="test"),
    )
    response = asyncio.run(chat.chat_sse(request))
    body = asyncio.run(_stream_body(response))
    events = [
        json.loads(frame.removeprefix("data: "))
        for frame in body.split("\n\n")
        if frame.strip()
    ]

    assert response.media_type == "text/event-stream"
    assert [event["type"] for event in events] == [
        "thought",
        "tool_call",
        "tool_result",
        "text",
        "done",
    ]
    assert FakeAgent.invocation is not None
    assert FakeAgent.invocation["message"] == request.message
    assert FakeAgent.invocation["history"] == [{"role": "assistant", "content": "已有上下文"}]
    assert FakeAgent.invocation["kwargs"]["scenario"] == "route_plan"
    assert FakeAgent.invocation["kwargs"]["current_location"]["city"] == "杭州"
    assert memory.messages[-2:] == [
        {"role": "user", "content": request.message},
        {"role": "assistant", "content": "推荐西湖群山路线。"},
    ]
