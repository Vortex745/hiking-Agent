"""Incremental delivery must precede completion and preserve the evidence gate."""
import asyncio
from types import SimpleNamespace

import pytest

from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator
from test_run_orchestration_chat import _FakeMemory, _repo, _settings, SUBJECT_ID


EVIDENCE = [{"tool": "hiking_knowledge_search", "phase": "result", "ok": True,
             "evidence": {"has_chunks": True, "sources": []}}]


def test_first_text_arrives_before_kernel_finishes_and_replay_matches(tmp_path):
    async def scenario():
        release = asyncio.Event()
        calls = []

        async def kernel(**kwargs):
            calls.append(kwargs)
            await kwargs["on_text"]("先穿登山鞋。", EVIDENCE)
            await release.wait()
            await kwargs["on_text"]("再检查天气。", EVIDENCE)
            return "先穿登山鞋。再检查天气。", EVIDENCE

        repo = _repo(tmp_path)
        orch = RunOrchestrator(settings=_settings(), runs_repo=repo,
                               memory=_FakeMemory(), kernel_turn=kernel)
        req = ChatFlowRequest(message="徒步建议", request_id="stream-first")
        gen = orch.stream_chat(req, subject_id=SUBJECT_ID)
        events = []
        try:
            while not events or events[-1].type != "text":
                # Cold tool imports are outside this ordering contract. The
                # unreleased barrier below proves delivery precedes completion.
                events.append(await asyncio.wait_for(anext(gen), timeout=10))
            assert not release.is_set()
            assert events[-1].payload["content"] == "先穿登山鞋。"
            stored = repo.get_events_since(run_id=events[-1].run_id, from_seq=0)
            assert stored[-1]["payload"]["content"] == "先穿登山鞋。"
            release.set()
            events.extend([event async for event in gen])
        finally:
            release.set()
            await gen.aclose()
        assert "".join(e.payload["content"] for e in events if e.type == "text") == "先穿登山鞋。再检查天气。"
        assert [e.type for e in events if e.type in {"done", "error", "cancelled"}] == ["done"]
        replay = [e async for e in orch.stream_chat(req, subject_id=SUBJECT_ID)]
        assert [(e.type, e.payload) for e in replay] == [(e.type, e.payload) for e in events]
        assert len(calls) == 1
    asyncio.run(scenario())


@pytest.mark.parametrize("history", [None, [
    {"role": "user", "content": "10月20日出发，两个人"},
    {"role": "assistant", "content": "已记录日期和人数。"},
]])
def test_kernel_uses_real_graph_tokens_and_filters_split_reasoning(monkeypatch, history):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain.agents import create_agent
    from agent_v2 import kernel as module

    model = FakeListChatModel(responses=["<think>private chain</think> 徒步安全第一。 "], sleep=0.001)

    graph = create_agent(model)

    class Agent:
        closed = False
        inputs = None

        def astream(self, *args, **kwargs):
            self.inputs = args[0]
            return graph.astream(*args, **kwargs)

        async def aclose(self):
            self.closed = True

    agent = Agent()

    async def assemble(*args):
        return agent

    monkeypatch.setattr(module, "assemble_kernel_agent", assemble)
    async def scenario():
        chunks = []
        thoughts = []
        async def on_text(text, audits):
            chunks.append(text)
        async def on_thought(text):
            thoughts.append(text)
        final, _ = await module.run_chat_turn(system_prompt="test", message="hi",
            settings=SimpleNamespace(), on_text=on_text, on_thought=on_thought,
            history=history)
        assert len(chunks) > 1
        assert "".join(chunks) == final == "徒步安全第一。"
        assert "".join(thoughts) == "private chain"
        expected = [(entry["role"], entry["content"]) for entry in history or []]
        assert agent.inputs == {"messages": expected + [("user", "hi")]}
        assert agent.closed
    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["cancel", "error"])
def test_partial_stream_has_one_terminal_and_no_late_text(tmp_path, outcome):
    from agent_v2.kernel import KernelModelFailure

    async def scenario():
        repo = _repo(tmp_path)
        release = asyncio.Event()

        async def kernel(**kwargs):
            await kwargs["on_text"]("已交付", EVIDENCE)
            await release.wait()
            if outcome == "error":
                raise KernelModelFailure("model_timeout", "超时")
            await kwargs["on_text"]("取消后不可交付", EVIDENCE)
            return "已交付取消后不可交付", EVIDENCE

        orch = RunOrchestrator(settings=_settings(), runs_repo=repo,
                               memory=_FakeMemory(), kernel_turn=kernel)
        gen = orch.stream_chat(ChatFlowRequest(message="徒步", request_id=outcome), subject_id=SUBJECT_ID)
        events = []
        try:
            while not events or events[-1].type != "text":
                events.append(await asyncio.wait_for(anext(gen), 2))
            if outcome == "cancel":
                repo.request_cancel(run_id=events[-1].run_id, idempotency_key="cancel-stream")
            release.set()
            events.extend([e async for e in gen])
        finally:
            release.set()
            await gen.aclose()
        assert [e.type for e in events if e.type in {"done", "error", "cancelled"}] == ["cancelled" if outcome == "cancel" else "error"]
        assert "".join(e.payload["content"] for e in events if e.type == "text") == "已交付"
        assert [e.sequence for e in events] == list(range(1, len(events) + 1))
    asyncio.run(scenario())


def test_invalid_tool_arguments_after_preamble_is_one_error_terminal(tmp_path):
    """A streamed preamble must not survive as a successful assistant answer."""
    from agent_v2.kernel import KernelModelFailure

    async def scenario():
        async def kernel(**kwargs):
            await kwargs["on_text"]("参数已确认，开始生成图片。", EVIDENCE)
            raise KernelModelFailure(
                "invalid_tool_arguments",
                "工具参数格式错误，本次请求未完成。",
            )

        repo = _repo(tmp_path)
        orch = RunOrchestrator(settings=_settings(), runs_repo=repo,
                               memory=_FakeMemory(), kernel_turn=kernel)
        request = ChatFlowRequest(
            message="生成图片",
            request_id="invalid-args-stream",
            conversation_id="conv-invalid-args",
        )
        events = [event async for event in orch.stream_chat(request, subject_id=SUBJECT_ID)]

        assert [e.type for e in events if e.type in {"done", "error", "cancelled"}] == ["error"]
        assert events[-1].payload["code"] == "invalid_tool_arguments"
        assert "".join(
            event.payload["content"] for event in events if event.type == "text"
        ) == "参数已确认，开始生成图片。"
        # The partial preamble may remain in the transcript, but the run and
        # its terminal are error — never a successful assistant completion.
        run = repo.control.get_run_by_request("invalid-args-stream")
        assert run.status == "error"
        stored = repo.checkpoints.get_messages(conversation_id="conv-invalid-args")
        assert [row.content for row in stored if row.role == "assistant"] == [
            "参数已确认，开始生成图片。"
        ]

    asyncio.run(scenario())


@pytest.mark.parametrize("content", ["<mm:think>private</mm:think>安全", "<think>private", "<reasoning>private</reasoning>安全"])
def test_reasoning_filter_matches_final_projection_at_every_split(content):
    from agent_v2.kernel import _StreamingPublicText, extract_final_text
    expected = extract_final_text({"messages": [SimpleNamespace(type="ai", content=content)]}) or ""
    for split in range(len(content) + 1):
        stream = _StreamingPublicText()
        actual = stream.feed(content[:split]) + stream.feed(content[split:]) + stream.feed("", final=True)
        assert actual == expected


def test_stream_does_not_leak_failed_or_not_yet_checked_search(tmp_path):
    async def scenario():
        failed = [{"tool": "hiking_knowledge_search", "phase": "result", "ok": True,
                   "evidence": {"has_chunks": False}}]

        async def kernel(**kwargs):
            await kwargs["on_text"]("未经检索的120公里", [])
            await kwargs["on_text"]("检索失败仍说120公里", failed)
            return "120公里", failed

        orch = RunOrchestrator(settings=_settings(), runs_repo=_repo(tmp_path),
                               memory=_FakeMemory(), kernel_turn=kernel)
        events = [e async for e in orch.stream_chat(
            ChatFlowRequest(message="路线里程", request_id="stream-gate"), subject_id=SUBJECT_ID)]
        text = "".join(e.payload["content"] for e in events if e.type == "text")
        assert text == "资料不足：未检索到可核验的徒步知识，无法提供具体徒步事实。"
    asyncio.run(scenario())


@pytest.mark.parametrize("profile_id", ["chat", "rag"])
@pytest.mark.parametrize("evidence_state", ["foreign_run", "no_search_tool"])
def test_stream_gate_uses_current_run_evidence_even_without_search_tool(
    tmp_path, monkeypatch, profile_id, evidence_state,
):
    from agent_v2 import kernel_tools

    if evidence_state == "no_search_tool":
        monkeypatch.setattr(kernel_tools, "build_chat_extra_tools", lambda *args, **kwargs: ())

    async def scenario():
        audits = [dict(EVIDENCE[0], run_id="another-run")] if evidence_state == "foreign_run" else []

        async def kernel(**kwargs):
            await kwargs["on_text"]("不应泄漏的1800米", audits)
            return "不应泄漏的1800米", audits

        repo = _repo(tmp_path)
        orch = RunOrchestrator(settings=_settings(), runs_repo=repo,
                               memory=_FakeMemory(), kernel_turn=kernel)
        req = ChatFlowRequest(message="路线爬升多少", request_id="stream-current-evidence",
                              profile_id=profile_id, conversation_id="conv-current-evidence")
        events = [event async for event in orch.stream_chat(req, subject_id=SUBJECT_ID)]
        text = "".join(event.payload["content"] for event in events if event.type == "text")
        assert text == "资料不足：未检索到可核验的徒步知识，无法提供具体徒步事实。"
        assert [event.type for event in events if event.type in {"done", "error", "cancelled"}] == ["done"]
        replay = [event async for event in orch.stream_chat(req, subject_id=SUBJECT_ID)]
        assert [(event.type, event.payload) for event in replay] == [(event.type, event.payload) for event in events]
        stored = repo.checkpoints.get_messages(conversation_id=req.conversation_id)
        assert [row.content for row in stored if row.role == "assistant"] == [text]

    asyncio.run(scenario())


def test_disconnect_cancels_incremental_kernel(tmp_path):
    async def scenario():
        closed = asyncio.Event()

        async def kernel(**kwargs):
            try:
                await kwargs["on_text"]("第一段", EVIDENCE)
                await asyncio.Event().wait()
            finally:
                closed.set()

        orch = RunOrchestrator(settings=_settings(), runs_repo=_repo(tmp_path),
                               memory=_FakeMemory(), kernel_turn=kernel)
        gen = orch.stream_chat(ChatFlowRequest(message="徒步", request_id="stream-close"),
                               subject_id=SUBJECT_ID)
        try:
            while (await asyncio.wait_for(anext(gen), 2)).type != "text":
                pass
        finally:
            await gen.aclose()
        await asyncio.wait_for(closed.wait(), 2)
    asyncio.run(scenario())


def test_thought_events_precede_text_and_replay(tmp_path):
    async def scenario():
        async def kernel(**kwargs):
            await kwargs["on_thought"]("先确认季节与海拔。")
            await kwargs["on_thought"]("再核对雪线与撤退点。")
            await kwargs["on_text"]("建议携带冰爪。", EVIDENCE)
            return "建议携带冰爪。", EVIDENCE

        repo = _repo(tmp_path)
        orch = RunOrchestrator(settings=_settings(), runs_repo=repo,
                               memory=_FakeMemory(), kernel_turn=kernel)
        req = ChatFlowRequest(message="雪线徒步", request_id="stream-thought")
        events = [event async for event in orch.stream_chat(req, subject_id=SUBJECT_ID)]
        types = [event.type for event in events]
        assert "thought" in types
        assert types.index("thought") < types.index("text")
        assert "".join(
            event.payload["content"] for event in events if event.type == "thought"
        ) == "先确认季节与海拔。再核对雪线与撤退点。"
        replay = [event async for event in orch.stream_chat(req, subject_id=SUBJECT_ID)]
        assert [(event.type, event.payload) for event in replay] == [
            (event.type, event.payload) for event in events
        ]

    asyncio.run(scenario())


def test_thoughts_can_be_disabled_without_touching_text(tmp_path):
    async def scenario():
        seen = {}

        async def kernel(**kwargs):
            seen["on_thought"] = kwargs.get("on_thought")
            await kwargs["on_text"]("正常回答。", EVIDENCE)
            return "正常回答。", EVIDENCE

        orch = RunOrchestrator(
            settings=_settings(agent_v2_thoughts_enabled=False),
            runs_repo=_repo(tmp_path),
            memory=_FakeMemory(),
            kernel_turn=kernel,
        )
        events = [event async for event in orch.stream_chat(
            ChatFlowRequest(message="徒步", request_id="stream-thought-off"),
            subject_id=SUBJECT_ID,
        )]
        assert seen["on_thought"] is None
        assert [event.type for event in events if event.type == "thought"] == []
        assert "".join(
            event.payload["content"] for event in events if event.type == "text"
        ) == "正常回答。"

    asyncio.run(scenario())
