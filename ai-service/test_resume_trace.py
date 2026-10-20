"""Resume stream tracing (ticket 05).

A resume is a separate HTTP execution: it always opens a new root trace
(``chat.resume``) and links back to the first stream only through ``run.id``
and the session attribute. First-stream nodes never merge into it.
"""

from types import SimpleNamespace

import pytest

import api.chat_v2 as chat_v2
from observability import tracing


class _Leases:
    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


class _Orchestrator:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.leases = _Leases()

    async def acquire_resume_leases(self, **_kwargs):
        return self.leases

    async def stream_resume(self, **_kwargs):
        if self.fail:
            raise RuntimeError("resume exploded")
        yield "data: first\n\n"
        yield "data: done\n\n"


def _request(runtime=None):
    app = SimpleNamespace(state=SimpleNamespace(agent_v2_runtime=runtime))
    return SimpleNamespace(app=app, headers={}, state=SimpleNamespace())


@pytest.fixture()
def spans(monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "_initialized", True)
    monkeypatch.setattr(tracing, "_shutdown", False)
    monkeypatch.setattr(tracing, "_init_failed", False)
    monkeypatch.setattr(tracing, "_provider", provider)
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer("test"))
    monkeypatch.setattr(tracing, "tracing_enabled", lambda: True)
    return exporter


def _call_resume(monkeypatch, orchestrator, *, run_id="run-1"):
    run = SimpleNamespace(
        request_id="req-1",
        conversation_id="conv-1",
        status="running",
    )
    monkeypatch.setattr(chat_v2, "_get_runs_repo", lambda _request: SimpleNamespace(
        get_run=lambda _run_id: run
    ))
    monkeypatch.setattr(chat_v2, "_require_conversation_owner", lambda *a, **k: None)
    monkeypatch.setattr(chat_v2, "_build_run_orchestrator", lambda _runtime: orchestrator)
    monkeypatch.setattr(chat_v2, "_sse_frame", lambda item: item)

    response = __import__("asyncio").run(
        chat_v2.chat_v2_resume(run_id, _request(), from_seq=0)
    )

    async def _drain():
        return [frame async for frame in response.body_iterator]

    return __import__("asyncio").run(_drain()), orchestrator


def _resume_spans(exporter):
    return [s for s in exporter.get_finished_spans() if s.name == "chat.resume"]


def test_resume_opens_a_new_root_trace(spans, monkeypatch):
    # Inside an active outer span (the first stream's context), the resume
    # still starts a brand-new trace instead of nesting under it.
    outer_trace_id = None
    with tracing.start_span("chat.request") as outer:
        outer_trace_id = outer.get_span_context().trace_id
        frames, orchestrator = _call_resume(monkeypatch, _Orchestrator())
        assert orchestrator.leases.released is True

    assert len(frames) == 2
    resume = _resume_spans(spans)[0]
    assert resume.parent is None
    assert resume.context.trace_id != outer_trace_id
    assert resume.attributes["run.id"] == "run-1"
    assert resume.attributes["request.id"] == "req-1"
    assert resume.attributes["conversation.id"] == "conv-1"
    assert resume.attributes["langfuse.session.id"] == "conv-1"


def test_two_resumes_of_same_run_get_distinct_traces(spans, monkeypatch):
    _call_resume(monkeypatch, _Orchestrator())
    _call_resume(monkeypatch, _Orchestrator())

    resumes = _resume_spans(spans)
    assert len(resumes) == 2
    assert resumes[0].context.trace_id != resumes[1].context.trace_id
    for span_ in resumes:
        assert span_.parent is None
        assert span_.attributes["run.id"] == "run-1"


def test_server_derived_conversation_id_falls_back_to_request_id():
    # The first stream has no client conversation id, so the server derives
    # ``conv-<request_id>``. A resume must land in the same session instead
    # of splitting the run across two Langfuse sessions.
    assert chat_v2._session_key(None, "req-1") == "req-1"
    assert chat_v2._session_key("conv-req-1", "req-1") == "req-1"
    assert chat_v2._session_key("conv-1", "req-1") == "conv-1"


def test_resume_failure_marks_span_and_still_releases(spans, monkeypatch):
    orchestrator = _Orchestrator(fail=True)
    with pytest.raises(RuntimeError):
        _call_resume(monkeypatch, orchestrator)

    assert orchestrator.leases.released is True
    resume = _resume_spans(spans)[0]
    assert resume.status.status_code.name == "ERROR"
    assert resume.attributes["error.type"] == "RuntimeError"
