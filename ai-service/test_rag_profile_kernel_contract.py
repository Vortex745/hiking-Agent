"""Spec #13 phase 2 (ticket #17) — rag profile kernel contract tests.

External behavior asserted:

* ``resolve_profile("rag")`` yields the knowledge-search-only surface,
  no MCP, no memory tools.
* The rag profile dispatches the shared kernel turn with the RAG system
  prompt and exactly one tool (the knowledge search).
* The grounding fail-closed gate covers the rag profile: a search that
  was attempted without usable chunks replaces the output; real evidence
  preserves it.
* The knowledge tool's embedded retrieve subgraph returns traceable
  chunks from a seeded SQLite store (deterministic fake embeddings, no
  network).

Same in-process harness as ``test_run_orchestration_chat.py``: fake
kernel entry + real SQLite persistence under a temp directory.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_v2.kernel import KernelModelFailure
from agent_v2.observability import CapacityExceeded
from agent_v2.profiles import ProfileId, is_tool_allowed, resolve_profile
from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator
from agent_v2.runs import RunRepository
from agent_v2.persistence import CheckpointsStore, ControlStore


TERMINALS = {"error", "done", "cancelled"}

SUBJECT_ID = "sub-rag"

_RAG_EVIDENCE_OK = {
    "tool": "hiking_knowledge_search",
    "phase": "result",
    "ok": True,
    "evidence": {"has_chunks": True, "sources": ["鳌太线攻略.md"]},
}

_RAG_EVIDENCE_EMPTY = {
    "tool": "hiking_knowledge_search",
    "phase": "result",
    "ok": True,
    "evidence": {"has_chunks": False, "sources": []},
}


def _repo(tmp_path: Path) -> RunRepository:
    return RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )


def _settings(**overrides: Any) -> SimpleNamespace:
    base = {
        "agent_v2_enabled": True,
        "openai_api_key": "sk-test",
        "openai_model": "test-model",
        "agent_v2_cowork_policy": "disabled",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _orchestrator(
    tmp_path: Path,
    kernel_calls: list[dict[str, Any]],
    *,
    settings: Any | None = None,
    kernel_error: Exception | None = None,
    kernel_result: tuple[str, list[dict[str, Any]]] | None = None,
) -> RunOrchestrator:
    async def fake_kernel(**kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
        kernel_calls.append(kwargs)
        if kernel_error is not None:
            raise kernel_error
        return kernel_result or ("根据知识库证据：及时补充电解质。", [])

    return RunOrchestrator(
        settings=settings or _settings(),
        runs_repo=_repo(tmp_path),
        memory=None,
        kernel_turn=fake_kernel,
        capacity=None,
    )


def _stream(orch: RunOrchestrator, req: ChatFlowRequest) -> list[Any]:
    async def run() -> list[Any]:
        return [item async for item in orch.stream_chat(req, subject_id=SUBJECT_ID)]

    return asyncio.run(run())


def _rag_req(message: str, request_id: str = "req-rag") -> ChatFlowRequest:
    return ChatFlowRequest(message=message, request_id=request_id, profile_id="rag")


# ---- profile surface -------------------------------------------------------


def test_rag_profile_resolves_with_knowledge_search_only() -> None:
    profile = resolve_profile("rag")
    assert profile.profile_id == ProfileId.RAG
    assert profile.allowed_tools == ("hiking_knowledge_search",)
    assert profile.allowed_mcp_servers is None
    assert is_tool_allowed(profile, "hiking_knowledge_search")
    assert not is_tool_allowed(profile, "gear_checklist")
    assert not is_tool_allowed(profile, "memory_save")


# ---- kernel-turn dispatch ---------------------------------------------------


def test_rag_profile_streams_kernel_turn_with_rag_prompt(tmp_path: Path) -> None:
    kernel_calls: list[dict[str, Any]] = []
    orch = _orchestrator(tmp_path, kernel_calls)

    items = _stream(orch, _rag_req("鳌太线夏季需要注意什么？"))

    types = [item.type for item in items]
    assert types == ["run_started", "progress", "heartbeat", "text", "done"]
    terminals = [t for t in types if t in TERMINALS]
    assert terminals == ["done"]

    call = kernel_calls[0]
    assert "hiking_knowledge_search" in call["system_prompt"]
    assert "来源" in call["system_prompt"]
    assert "知识库中暂无相关资料" in call["system_prompt"]
    assert call["message"] == "鳌太线夏季需要注意什么？"

    tool_names = sorted(tool.name for tool in call["extra_tools"])
    assert tool_names == ["hiking_knowledge_search"]


def test_rag_profile_replaces_output_when_search_yields_no_evidence(tmp_path: Path) -> None:
    kernel_calls: list[dict[str, Any]] = []
    orch = _orchestrator(
        tmp_path,
        kernel_calls,
        kernel_result=(
            "鳌太线全程约120公里，冬季水源很少。",
            [_RAG_EVIDENCE_EMPTY],
        ),
    )

    items = _stream(orch, _rag_req("查鳌太线冬季精确里程", request_id="req-rag-empty"))

    text = next(event.payload["content"] for event in items if event.type == "text")
    assert text == "资料不足：未检索到可核验的徒步知识，无法提供具体徒步事实。"
    assert "120公里" not in text


def test_rag_profile_preserves_answer_when_knowledge_evidence_exists(tmp_path: Path) -> None:
    expected = "根据知识库证据：鳌太线最佳窗口为6-9月（来源：鳌太线攻略.md）。"
    orch = _orchestrator(
        tmp_path,
        [],
        kernel_result=(expected, [_RAG_EVIDENCE_OK]),
    )

    items = _stream(orch, _rag_req("鳌太线什么季节走合适？", request_id="req-rag-ok"))

    text = next(event.payload["content"] for event in items if event.type == "text")
    assert text == expected


# ---- embedded retrieve subgraph (knowledge tool) ----------------------------


class _DeterministicQueryEmbedding:
    """Offline stand-in for the LangChain embeddings used by the retriever."""

    def __init__(self, dimensions: int = 1536):
        self._dimensions = dimensions

    def _vec(self, text: str) -> list[float]:
        vec = [(ord(ch) % 7 + 1) * 0.1 for ch in text[:12]]
        while len(vec) < self._dimensions:
            vec.append(0.0)
        return vec[: self._dimensions]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


@pytest.fixture()
def _seeded_knowledge_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Seed a temp SQLite store and point the tool's stack at it."""
    from config import settings
    from rag.ingestion import FakeEmbedding, IngestionPipeline
    from rag.sqlite_store import SQLiteStore

    db_path = tmp_path / "rag.sqlite"
    monkeypatch.setattr(settings, "rag_sqlite_db_path", str(db_path))

    store = SQLiteStore(str(db_path), embedding_dimensions=1536)
    pipeline = IngestionPipeline(store=store, embedding_provider=FakeEmbedding(dimensions=1536))
    report = pipeline.ingest_content(
        markdown="# 鳌太线装备\n\n鳌太线穿越需要帐篷、睡袋和防风外套，夏季仍需保暖。",
        source="鳌太线攻略.md",
        file_name="鳌太线攻略.md",
        status="active",
    )
    assert report.child_count > 0

    import tools.hiking_knowledge as hk

    saved = (hk._retriever, hk._subgraph)
    hk._retriever = None
    hk._subgraph = None
    yield hk
    hk._retriever, hk._subgraph = saved


def test_knowledge_tool_returns_traceable_chunks_via_embedded_subgraph(
    _seeded_knowledge_store: Any,
) -> None:
    import tools.hiking_knowledge as hk

    retriever, _subgraph = hk._get_retrieval_stack()
    # Offline: replace the LangChain embeddings with a deterministic stand-in
    # (the seeded store's vectors are also deterministic fake embeddings).
    retriever.embeddings = _DeterministicQueryEmbedding(
        dimensions=settings_embedding_dimensions()
    )

    result = asyncio.run(
        hk.hiking_knowledge_search.ainvoke({"query": "鳌太线需要带什么装备？"})
    )

    assert result["chunks"], f"expected chunks, got: {result}"
    first = result["chunks"][0]
    assert first["source"] == "鳌太线攻略.md"
    assert first["preview"]
    assert "找到" in result["message"]


def settings_embedding_dimensions() -> int:
    from config import settings

    return settings.embedding_dimensions
