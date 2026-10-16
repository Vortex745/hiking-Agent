"""harness-p0-p1 Ticket 04 — real PDF rendering + illegal artifact rejection.

Spec coverage: T05 (rendering) + T06 (artifacts/download).

The old artifact path wrote the draft as a TEXT file with a ``.pdf``
suffix and published it after a manifest checksum — a "PDF" that no
reader can open. This contract pins the fixed behavior:

* the approved export renders through the shared generic renderer into
  REAL PDF bytes: Chinese/English/special characters, headings,
  paragraphs, lists, tables, A4 pagination with repeated table headers
  and page numbers all survive a real parser (pypdf);
* a blank body is rejected — never a "successful" header-only file;
* HTML in the draft is escaped text: no scripts, links or images are
  executed, fetched or embedded; the route-facts-pending notice always
  rides along; the CID font fallback keeps minimal environments working;
* publish and download share the same content contract: text-fake PDFs,
  truncated/corrupted files, encrypted files and page-less documents are
  rejected; legacy illegal files do not download, and neither file nor
  record is deleted;
* the approval flow generates at most once, never publishes on failure,
  keeps cancel boundaries, and the placeholder remote sandbox still
  refuses (Ticket 01) — the rendering fix does not resurrect host
  execution.

Run (ai-service working directory):
    E:/miniconda3/python.exe -B -m pytest -q test_pdf_bytes_contract.py
"""

from __future__ import annotations

import asyncio
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_v2.computer import (
    LocalNativeComputer,
    RemoteE2BComputer,
    SandboxUnavailableError,
    create_computer,
)
from agent_v2.artifacts import (
    ArtifactExportError,
    export_artifact_to_host,
    validate_pdf_bytes,
)
from agent_v2.pdf_rendering import (
    PDFRenderingError,
    render_markdown_pdf_bytes,
)
from agent_v2.runs import RunRepository
from agent_v2.persistence import CheckpointsStore, ControlStore


DRAFT = """# 杭州两日徒步计划

这是一个通用草案，包含中英文混排与特殊字符：Café — 25% & <tag> "quoted"。

## 每日安排

- 第一天：沿溪谷上行，速率保守。
- 第二天：原路下撤，避免夜行。

1. 核对天气预警
2. 确认下撤点

## 装备表

| 类别 | 物品 | 数量 |
|---|---|---|
| 基础 | 登山鞋 | 1 |
| 安全 | 急救包 | 1 |
"""

_LONG_TABLE_DRAFT = "# 长表分页\n\n| 页 | 内容 |\n|---|---|\n" + "".join(
    f"| {i} | 第 {i} 行内容，用于跨页与表头重复验证 |\n" for i in range(1, 61)
)

_HTML_DRAFT = """# HTML 注入样本

<script>alert("xss")</script>

<img src="http://evil.invalid/x.png" onerror="alert(1)">

<a href="http://evil.invalid/steal">点我</a>

段落正文：<b>加粗标签</b>与 &amp; 实体。
"""


def _extract_text(pdf_bytes: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(pdf_bytes))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


# ---- rendering: real PDF ------------------------------------------------------


def test_render_produces_real_parseable_pdf_with_mixed_text():
    data = render_markdown_pdf_bytes(DRAFT)

    assert data.startswith(b"%PDF")
    assert data.rstrip().endswith(b"%%EOF")
    validate_pdf_bytes(data)  # shared content contract accepts it

    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    assert len(reader.pages) >= 1
    page = reader.pages[0]
    box = page.mediabox
    # A4: 595.27 x 841.89 pt.
    assert abs(float(box.width) - 595.27) < 1.0
    assert abs(float(box.height) - 841.89) < 1.0

    text = _extract_text(data)
    assert "杭州两日徒步计划" in text
    assert "Café" in text
    assert "25%" in text
    assert "每日安排" in text


def test_render_covers_headings_paragraphs_lists_and_tables():
    data = render_markdown_pdf_bytes(DRAFT)
    text = _extract_text(data)

    assert "每日安排" in text          # heading
    assert "沿溪谷上行" in text        # bullet paragraph
    assert "核对天气预警" in text      # ordered list
    assert "装备表" in text            # table section heading
    assert "登山鞋" in text            # table cell
    assert "急救包" in text


def test_render_paginates_with_page_numbers_and_repeated_table_header():
    data = render_markdown_pdf_bytes(_LONG_TABLE_DRAFT)

    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    assert len(reader.pages) >= 2, "long table must paginate"
    for index, page in enumerate(reader.pages, 1):
        page_text = page.extract_text() or ""
        assert f"第 {index} 页" in page_text          # footer page number
        assert "页" in page_text and "内容" in page_text  # header repeats


def test_blank_body_is_rejected():
    for blank in ("", "   ", "\n\t  \n"):
        with pytest.raises(PDFRenderingError):
            render_markdown_pdf_bytes(blank)


def test_html_in_body_is_escaped_never_executed():
    data = render_markdown_pdf_bytes(_HTML_DRAFT)

    text = _extract_text(data)
    # The raw markup survives as literal TEXT…
    assert "<script>" in text
    assert "evil.invalid" in text
    # …and nothing is executable: no link annotations, no images.
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    for page in reader.pages:
        annotations = page.get("/Annots")
        if annotations:
            for annotation in annotations:
                annot = annotation.get_object()
                assert "/URI" not in annot, "link annotation leaked into the PDF"
        resources = page.get("/Resources")
        if resources:
            xobjects = resources.get_object().get("/XObject")
            if xobjects:
                for name in xobjects.get_object():
                    assert xobjects.get_object()[name].get_object().get("/Subtype") != "/Image"


def test_route_facts_notice_always_present():
    text = _extract_text(render_markdown_pdf_bytes(DRAFT))
    assert "路线事实待核验" in text


def test_cid_fallback_renders_chinese_in_minimal_font_env(tmp_path, monkeypatch):
    import agent_v2.pdf_rendering as rendering

    monkeypatch.delenv("AI_HIKING_PDF_FONT", raising=False)
    monkeypatch.setattr(rendering, "_FONT_DIRS", [str(tmp_path / "no-fonts-here")])
    regular, bold = rendering.register_pdf_font()
    # The deterministic fallback chain — never a hard failure.
    assert regular in {"STSong-Light", "Helvetica"}

    data = rendering.render_markdown_pdf_bytes("# 中文字体回退\n\n最小字体环境下的中文正文。")
    validate_pdf_bytes(data)
    assert "中文字体回退" in _extract_text(data)


# ---- trusted-local generation --------------------------------------------------


def test_local_native_generate_pdf_writes_real_pdf_bytes(tmp_path):
    computer = LocalNativeComputer()
    workspace = computer.create_workspace("run-pdf-1")
    try:
        relative = computer.generate_pdf(workspace, DRAFT, "hiking-plan.pdf")
        assert relative == "hiking-plan.pdf"
        data = computer.read_artifact_bytes(workspace, relative)
        validate_pdf_bytes(data)
        manifest = computer.build_manifest(workspace, relative, "application/pdf")
        assert manifest.sha256 and manifest.size_bytes == len(data)
    finally:
        computer.cleanup(workspace)


def test_local_native_generate_pdf_rejects_blank_body(tmp_path):
    computer = LocalNativeComputer()
    workspace = computer.create_workspace("run-pdf-blank")
    try:
        from agent_v2.computer import SandboxGenerationError

        with pytest.raises(SandboxGenerationError):
            computer.generate_pdf(workspace, "   \n ", "blank.pdf")
        # No pseudo file was written.
        assert not (Path(workspace) / "blank.pdf").exists()
    finally:
        computer.cleanup(workspace)


# ---- publish validation --------------------------------------------------------


class _FakeComputer:
    """Computer double serving canned artifact bytes."""

    sandbox_type = "local_native"

    def __init__(self, data: bytes, filename: str = "hiking-plan.pdf") -> None:
        self._data = data
        self._filename = filename
        self.commands: list[str] = []

    def create_workspace(self, run_id: str) -> str:
        return f"/ws/{run_id}"

    def build_manifest(self, workspace_path, relative_path, declared_mime):
        from agent_v2.artifacts import ArtifactManifest, compute_sha256

        return ArtifactManifest(
            relative_path=relative_path,
            public_filename=self._filename,
            declared_mime=declared_mime,
            size_bytes=len(self._data),
            sha256=compute_sha256(self._data),
        )

    def read_artifact_bytes(self, workspace_path, relative_path) -> bytes:
        return self._data

    def run_bash(self, *a, **k):
        raise AssertionError("publish validation never runs commands")


def _repo(tmp_path: Path) -> RunRepository:
    return RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )


def _run_row(repo: RunRepository, request_id: str) -> str:
    run_id, _ = repo.create_run_idempotent(
        request_id=request_id,
        conversation_id=f"conv-{request_id}",
        profile_id="artifact",
        user_message="规划杭州两日徒步并出 PDF 计划",
    )
    return run_id


def _export(tmp_path: Path, data: bytes, request_id: str, *, filename: str = "hiking-plan.pdf"):
    repo = _repo(tmp_path)
    run_id = _run_row(repo, request_id)
    computer = _FakeComputer(data, filename)
    storage = tmp_path / f"storage-{request_id}"
    try:
        result = export_artifact_to_host(
            computer=computer,
            workspace_path=f"/ws/{run_id}",
            relative_path=filename,
            declared_mime="application/pdf",
            storage_dir=str(storage),
            runs_repo=repo,
            run_id=run_id,
            conversation_id=f"conv-{request_id}",
        )
    except ArtifactExportError as exc:
        return repo, run_id, None, exc
    return repo, run_id, result, None


def test_export_accepts_real_pdf(tmp_path):
    data = render_markdown_pdf_bytes(DRAFT)
    repo, run_id, result, error = _export(tmp_path, data, "req-pdf-ok")

    assert error is None, error
    assert result["mime_type"] == "application/pdf"
    assert result["size_bytes"] == len(data)
    artifacts = repo.list_artifacts_for_run(run_id)
    assert artifacts and artifacts[0].status == "available"


def test_export_rejects_text_fake_pdf(tmp_path):
    fake = "这不是 PDF，只是穿了 .pdf 外衣的文本。".encode("utf-8")
    repo, run_id, result, error = _export(tmp_path, fake, "req-pdf-fake")

    assert result is None
    assert error is not None and "%PDF" in str(error)
    artifacts = repo.list_artifacts_for_run(run_id)
    assert artifacts and artifacts[0].status == "failed"
    assert artifacts[0].failure_reason and "pdf" in artifacts[0].failure_reason.lower()


def test_export_rejects_truncated_pdf(tmp_path):
    data = render_markdown_pdf_bytes(DRAFT)
    repo, run_id, result, error = _export(tmp_path, data[:-80], "req-pdf-trunc")

    assert result is None
    assert error is not None
    artifacts = repo.list_artifacts_for_run(run_id)
    assert artifacts and artifacts[0].status == "failed"


def test_export_rejects_encrypted_pdf(tmp_path):
    encrypted = _encrypted_pdf_bytes()
    repo, run_id, result, error = _export(tmp_path, encrypted, "req-pdf-enc")

    assert result is None
    assert error is not None and "加密" in str(error)
    artifacts = repo.list_artifacts_for_run(run_id)
    assert artifacts and artifacts[0].status == "failed"


def test_export_rejects_pageless_pdf(tmp_path):
    pageless = b"%PDF-1.4\n%%EOF\n"
    repo, run_id, result, error = _export(tmp_path, pageless, "req-pdf-nopages")

    assert result is None
    assert error is not None
    artifacts = repo.list_artifacts_for_run(run_id)
    assert artifacts and artifacts[0].status == "failed"


def _encrypted_pdf_bytes() -> bytes:
    """A genuinely encrypted PDF (pypdf writer, empty user password)."""
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(BytesIO(render_markdown_pdf_bytes(DRAFT)))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt(user_password="", owner_password="owner-secret")
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


# ---- download validation -------------------------------------------------------


def _artifact_app(tmp_path: Path, repo: RunRepository):
    from fastapi import FastAPI

    import api.chat_v2 as chat_v2_module

    class _FakeMemory:
        def resolve_subject(self, cookie):
            return SimpleNamespace(subject_id="sub-pdf", issued=False, token=None)

        def owns_conversation(self, subject_id, conversation_id):
            return conversation_id.startswith("conv-")

    class _Runtime:
        runs = repo
        memory = _FakeMemory()
        telemetry = None
        capacity = None
        kernel_checkpoints = None

    app = FastAPI()
    app.include_router(chat_v2_module.artifacts_v2_router, prefix="/api/v2")
    app.state.agent_v2_runtime = _Runtime()
    return app


def _insert_artifact_row(
    repo: RunRepository,
    tmp_path: Path,
    *,
    artifact_id: str,
    data: bytes,
    filename: str = "hiking-plan.pdf",
    mime: str = "application/pdf",
    status: str = "available",
) -> Path:
    from datetime import datetime, timedelta, timezone

    run_id = _run_row(repo, f"req-dl-{artifact_id[:8]}")
    storage = tmp_path / "dl-storage" / artifact_id
    storage.mkdir(parents=True, exist_ok=True)
    file_path = storage / filename
    file_path.write_bytes(data)
    now = datetime.now(timezone.utc)
    repo.insert_artifact(
        artifact_id=artifact_id,
        run_id=run_id,
        conversation_id=f"conv-req-dl-{artifact_id[:8]}",
        public_filename=filename,
        mime_type=mime,
        size_bytes=len(data),
        sha256="0" * 64,
        storage_path=str(file_path),
        status=status,
        failure_reason=None,
        created_at=now.isoformat(),
        expires_at=(now + timedelta(days=7)).isoformat(),
    )
    return file_path


async def _get(app, path: str):
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as client:
        return await client.get(path)


def test_download_serves_real_pdf(tmp_path):
    import uuid

    repo = _repo(tmp_path)
    data = render_markdown_pdf_bytes(DRAFT)
    artifact_id = str(uuid.uuid4())
    file_path = _insert_artifact_row(repo, tmp_path, artifact_id=artifact_id, data=data)
    app = _artifact_app(tmp_path, repo)

    resp = asyncio.run(_get(app, f"/api/v2/artifacts/{artifact_id}"))

    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/pdf")
    assert resp.content == data
    assert file_path.is_file()


def test_download_rejects_legacy_fake_pdf_without_deleting_evidence(tmp_path):
    import uuid

    repo = _repo(tmp_path)
    fake = b"legacy text pretending to be a pdf"
    artifact_id = str(uuid.uuid4())
    file_path = _insert_artifact_row(repo, tmp_path, artifact_id=artifact_id, data=fake)

    before = repo.get_artifact(artifact_id)
    assert before is not None and before.status == "available"

    app = _artifact_app(tmp_path, repo)
    resp = asyncio.run(_get(app, f"/api/v2/artifacts/{artifact_id}"))

    assert resp.status_code == 410, resp.text
    # Evidence is preserved: neither the file nor the row is deleted.
    assert file_path.is_file()
    after = repo.get_artifact(artifact_id)
    assert after is not None and after.status == "available"
    assert after.storage_path == before.storage_path


def test_download_rejects_truncated_pdf_without_deleting_evidence(tmp_path):
    import uuid

    repo = _repo(tmp_path)
    data = render_markdown_pdf_bytes(DRAFT)[:-60]
    artifact_id = str(uuid.uuid4())
    file_path = _insert_artifact_row(repo, tmp_path, artifact_id=artifact_id, data=data)

    app = _artifact_app(tmp_path, repo)
    resp = asyncio.run(_get(app, f"/api/v2/artifacts/{artifact_id}"))

    assert resp.status_code == 410
    assert file_path.is_file()


def test_download_keeps_ownership_and_expiry_rules(tmp_path):
    import uuid
    from datetime import datetime, timedelta, timezone

    repo = _repo(tmp_path)
    data = render_markdown_pdf_bytes(DRAFT)
    artifact_id = str(uuid.uuid4())
    file_path = _insert_artifact_row(repo, tmp_path, artifact_id=artifact_id, data=data)

    # Cross-conversation access stays forbidden.
    app = _artifact_app(tmp_path, repo)
    resp = asyncio.run(
        _get(app, f"/api/v2/artifacts/{artifact_id}?conversation_id=someone-else")
    )
    assert resp.status_code == 403

    # Expiry stays enforced (410) and the file survives.
    expired_at = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    repo.control._conn.execute(
        "UPDATE artifacts SET expires_at = ? WHERE artifact_id = ?",
        (expired_at, artifact_id),
    )
    resp = asyncio.run(_get(app, f"/api/v2/artifacts/{artifact_id}"))
    assert resp.status_code == 410
    assert file_path.is_file()


# ---- remote placeholder still refuses (Ticket 01) ------------------------------


def test_remote_sandbox_still_refuses_pdf_generation():
    computer = create_computer("remote_e2b")
    with pytest.raises(SandboxUnavailableError):
        computer.generate_pdf("/ws/x", DRAFT, "plan.pdf")


# ---- approval flow: generate once, never publish on failure --------------------


def _settings(**overrides: Any) -> SimpleNamespace:
    base = {
        "agent_v2_enabled": True,
        "openai_api_key": "sk-test",
        "openai_model": "test-model",
        "agent_v2_state_dir": "state",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _pause_flow(orch, repo: RunRepository, request_id: str) -> tuple[str, str]:
    from agent_v2.run_orchestration import ChatFlowRequest

    req = ChatFlowRequest(
        message="规划杭州两日徒步并出 PDF 计划",
        request_id=request_id,
        profile_id="artifact",
    )

    async def run() -> list[Any]:
        return [item async for item in orch.stream_chat(req, subject_id="sub-pdf")]

    items = asyncio.run(run())
    types = [e.type for e in items]
    assert types[-1] == "run.paused", types
    run_row = repo.control.get_run_by_request(request_id)
    approval = repo.list_pending_approvals_for_run(run_row.run_id)[0]
    return run_row.run_id, approval.approval_id


def _orchestrator(tmp_path: Path, repo: RunRepository, computer, cleanups, *, plan_text: str = DRAFT):
    from agent_v2.run_orchestration import RunOrchestrator

    async def fake_kernel(**kwargs: Any):
        return (plan_text, [])

    return RunOrchestrator(
        settings=_settings(agent_v2_state_dir=str(tmp_path / "state")),
        runs_repo=repo,
        memory=SimpleNamespace(
            memory_enabled=lambda *a: True,
            claim_conversation=lambda *a, **k: True,
            owns_conversation=lambda *a, **k: True,
            create_candidate=lambda *a, **k: {},
        ),
        kernel_turn=fake_kernel,
        computer_factory=lambda *a, **k: computer,
        sandbox_cleanup=lambda *a, **k: cleanups.append(k),
        gate_factory=None,  # real gate → generate_pdf HIGH risk → pause
    )


def _stream_resume(orch, run_id: str, *, from_seq: int = 0) -> list[Any]:
    async def run() -> list[Any]:
        return [item async for item in orch.stream_resume(run_id=run_id, from_seq=from_seq)]

    return asyncio.run(run())


def test_approved_resume_generates_real_pdf_exactly_once(tmp_path):
    from agent_v2.computer import ComputerError  # noqa: F401

    repo = _repo(tmp_path)
    computer = LocalNativeComputer()
    cleanups: list[Any] = []
    orch = _orchestrator(tmp_path, repo, computer, cleanups)

    run_id, approval_id = _pause_flow(orch, repo, "req-pdf-approve")
    repo.resolve_approval(
        approval_id=approval_id, status="approved", decision_request_id="dec-pdf-approve"
    )

    items = _stream_resume(orch, run_id)
    types = [e.type for e in items]
    assert types[-1] == "done", types
    available = next(e for e in items if e.type == "artifact.available")
    artifact_id = available.payload["artifact_id"]

    # The published bytes are a real PDF.
    artifact = repo.get_artifact(artifact_id)
    assert artifact is not None and artifact.status == "available"
    data = Path(artifact.storage_path).read_bytes()
    validate_pdf_bytes(data)
    assert "杭州两日徒步计划" in _extract_text(data)

    # A second resume replays the stored stream — no second generation, no
    # second export, no duplicated events.
    replayed = _stream_resume(orch, run_id, from_seq=1)
    assert [e.type for e in replayed][-1] == "done"
    assert len([e for e in replayed if e.type == "artifact_candidate"]) == 1
    assert len([e for e in replayed if e.type == "artifact.available"]) == 1
    assert len([e for e in replayed if e.type == "tool_call"]) == 1
    artifacts = repo.list_artifacts_for_run(run_id)
    assert len(artifacts) == 1


def test_rejected_approval_never_generates(tmp_path):
    repo = _repo(tmp_path)
    computer = LocalNativeComputer()
    cleanups: list[Any] = []
    orch = _orchestrator(tmp_path, repo, computer, cleanups)

    run_id, approval_id = _pause_flow(orch, repo, "req-pdf-reject")
    repo.resolve_approval(
        approval_id=approval_id, status="rejected", decision_request_id="dec-pdf-reject"
    )

    items = _stream_resume(orch, run_id)
    types = [e.type for e in items]
    assert types[-1] == "done"
    assert "artifact_candidate" not in types
    assert repo.list_artifacts_for_run(run_id) == []


def test_generation_failure_never_publishes_artifact(tmp_path):
    from agent_v2.computer import SandboxGenerationError

    class _FailingGenerator(LocalNativeComputer):
        def generate_pdf(self, workspace_path, content, filename):
            raise SandboxGenerationError("injected rendering failure")

    repo = _repo(tmp_path)
    computer = _FailingGenerator()
    cleanups: list[Any] = []
    orch = _orchestrator(tmp_path, repo, computer, cleanups)

    run_id, approval_id = _pause_flow(orch, repo, "req-pdf-genfail")
    repo.resolve_approval(
        approval_id=approval_id, status="approved", decision_request_id="dec-pdf-genfail"
    )

    items = _stream_resume(orch, run_id)
    types = [e.type for e in items]
    assert types[-1] == "error", types
    assert "artifact.available" not in types
    # No available artifact was ever published.
    assert all(a.status != "available" for a in repo.list_artifacts_for_run(run_id))
