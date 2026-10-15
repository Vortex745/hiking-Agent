"""harness-p0-p1 Ticket 07 — failure injection proves the key regressions are real.

Spec coverage: T13 (ticket AC4: 通过安全替身注入已知旧缺陷时关键回归及 job
真实失败；语法编译不能替代行为测试).

The validation job must be a BEHAVIORAL gate: `python -m compileall` passed
on the defect-laden product, which is exactly why ticket 07 replaces it.
This contract injects the KNOWN PRE-FIX defect of ticket families 01/02/04/05
through SAFE sentinel doubles — recorded attempts only, no host command ever
executes, no network call is made — and asserts that the corresponding guard
(the distilled key assertion of that ticket's own contract suite) REALLY
fails on the defect while passing on the fixed code:

* 01 — the old ``RemoteE2BComputer`` ran the placeholder lifecycle on the
  HOST (``tempfile.mkdtemp`` + ``bash -c``); the guard is
  ``test_remote_computer_fail_closed.py::test_remote_computer_every_operation_refuses_before_host_access``;
* 02 — the old evidence gate only blocked the "search attempted without
  chunks" signature, so a never-searched turn passed arbitrary factual text
  through; the guard is
  ``test_chat_evidence_gate_contract.py::test_never_searched_factual_output_is_blocked``;
* 04 — the old artifact path published the draft as a TEXT stub named
  ``.pdf`` with no PDF content verification; the guards are
  ``test_pdf_bytes_contract.py::test_render_produces_real_parseable_pdf_with_mixed_text``
  and ``::test_export_rejects_text_fake_pdf``;
* 05 — the old store read was a naive raw dump (un-trimmed, unfinished runs
  included); the guard is
  ``test_conversation_context_contract.py::test_history_window_keeps_at_most_twenty_messages_by_whole_turns``.

How the CI job runs this: it is part of the full offline backend regression
(`cd ai-service && python -m pytest -q -rs`), so it gates every pull request.
`test_injected_defects_fail_a_real_pytest_job_while_syntax_stays_green`
additionally proves the JOB-level failure with a real pytest subprocess over
a generated scenario file — while ``compileall`` stays green on the same
file, i.e. syntax compilation can NOT substitute for behavior tests.

Run (ai-service working directory):
    E:/miniconda3/python.exe -B -m pytest -q test_regression_injection_contract.py
"""

from __future__ import annotations

import asyncio
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

AI_SERVICE_DIR = Path(__file__).resolve().parent
REPO_ROOT = AI_SERVICE_DIR.parent

UNSUPPORTED_FACT = "这段路线爬升约1800米。"
CONVERSATION = "conv-injection-1"

_DRAFT = (
    "# 杭州两日徒步计划\n\n"
    "这是一个通用草案，包含中英文混排与特殊字符：Café — 25% & <tag>。\n\n"
    "## 每日安排\n\n"
    "- 第一天：沿溪谷上行，速率保守。\n"
)

_TEXT_FAKE_PDF = "这不是 PDF，只是穿了 .pdf 外衣的文本。".encode("utf-8")


# ---------------------------------------------------------------------------
# shared helpers (mirrors of the contract suites' own helpers)
# ---------------------------------------------------------------------------


def _repo(workdir: Path, name: str = "") -> Any:
    from agent_v2.persistence import CheckpointsStore, ControlStore
    from agent_v2.runs import RunRepository

    workdir.mkdir(parents=True, exist_ok=True)
    return RunRepository(
        control=ControlStore(str(workdir / f"control{name}.sqlite")),
        checkpoints=CheckpointsStore(str(workdir / f"ckpt{name}.sqlite")),
    )


def _settings(**overrides: Any) -> Any:
    base = {
        "agent_v2_enabled": True,
        "openai_api_key": "sk-test",
        "openai_model": "test-model",
        "agent_v2_cowork_policy": "disabled",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeMemory:
    def memory_enabled(self, subject_id): return True

    def claim_conversation(self, s, c): return True

    def owns_conversation(self, s, c): return True

    def create_candidate(self, s, **kw): return kw


class _FakeArtifactComputer:
    """Computer double serving canned artifact bytes (no host access)."""

    sandbox_type = "local_native"

    def __init__(self, data: bytes, filename: str = "hiking-plan.pdf") -> None:
        self._data = data
        self._filename = filename

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


class _HostAttemptRecorder:
    """Records host-access ATTEMPTS without executing anything.

    This is the safety sentinel of the injection: the wrapped primitives
    (``tempfile.mkdtemp`` / ``subprocess`` / ``shutil.rmtree``) only append
    to :attr:`attempts` and return inert stubs. No temp directory is
    created, no shell command runs, no file is removed — the old defect is
    reproduced in behavior, never in effect.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.attempts: list[str] = []

        def rec_mkdtemp(*args: Any, **kwargs: Any) -> str:
            self.attempts.append("tempfile.mkdtemp")
            return str(Path(tempfile.gettempdir()) / "injected-never-created")

        def rec_run(command: Any, *args: Any, **kwargs: Any):
            self.attempts.append(f"subprocess.run {command!r}")
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="", stderr=""
            )

        def rec_popen(command: Any, *args: Any, **kwargs: Any):
            self.attempts.append(f"subprocess.Popen {command!r}")
            return SimpleNamespace(returncode=0)

        def rec_rmtree(path: Any, *args: Any, **kwargs: Any) -> None:
            self.attempts.append(f"shutil.rmtree {path!r}")

        monkeypatch.setattr(tempfile, "mkdtemp", rec_mkdtemp)
        monkeypatch.setattr(subprocess, "run", rec_run)
        monkeypatch.setattr(subprocess, "Popen", rec_popen)
        monkeypatch.setattr(shutil, "rmtree", rec_rmtree)


def _seed_turn(
    repo: Any,
    *,
    request_id: str,
    user_msg: str,
    assistant_msg: str,
    completed: bool,
    conversation_id: str = CONVERSATION,
) -> str:
    run_id, _ = repo.create_run_idempotent(
        request_id=request_id,
        conversation_id=conversation_id,
        profile_id="chat",
        user_message=user_msg,
    )
    repo.checkpoints.append_message(
        conversation_id=conversation_id,
        run_id=run_id,
        role="assistant",
        content=assistant_msg,
        sequence=2,
    )
    if completed:
        repo.mark_run_done(run_id)
    return run_id


# ---------------------------------------------------------------------------
# ticket 01 — remote placeholder must fail closed BEFORE any host access
# ---------------------------------------------------------------------------


def _inject_01_remote_host_stub(
    monkeypatch: pytest.MonkeyPatch, workdir: Path
) -> _HostAttemptRecorder:
    """Re-create the PRE-TICKET-01 ``RemoteE2BComputer``: the placeholder
    mimicked the E2B lifecycle ON THE HOST (temp dir + ``bash -c``).

    Every host access flows through the record-only sentinel; nothing runs.
    """
    from agent_v2.computer import RemoteE2BComputer

    recorder = _HostAttemptRecorder(monkeypatch)

    def create_workspace(self: Any, run_id: str) -> str:
        # Old behavior: host temp directory. Sentinel-recorded, never created.
        return tempfile.mkdtemp(prefix="v2-e2b-")

    def generate_pdf(self: Any, workspace_path: str, content: str, filename: str) -> str:
        return f"{workspace_path}/{filename}"

    def build_manifest(
        self: Any, workspace_path: str, relative_path: str, declared_mime: str
    ) -> Any:
        return SimpleNamespace(
            relative_path=relative_path,
            public_filename=Path(relative_path).name,
            declared_mime=declared_mime,
            size_bytes=0,
            sha256="0" * 64,
        )

    def read_artifact_bytes(self: Any, workspace_path: str, relative_path: str) -> bytes:
        return b""

    def run_bash(
        self: Any,
        workspace_path: str,
        command: str,
        *,
        timeout_seconds: float = 30.0,
        max_output_bytes: int = 1_048_576,
    ) -> Any:
        # Old behavior: host ``bash -c``. ``subprocess`` is sentinel-wrapped,
        # so this records the attempt and returns an inert stub.
        completed = subprocess.run(["bash", "-c", command], capture_output=True)
        return SimpleNamespace(returncode=completed.returncode, stdout=b"", stderr=b"")

    def cleanup(self: Any, workspace_path: str) -> None:
        shutil.rmtree(workspace_path, ignore_errors=True)

    for name, replacement in (
        ("create_workspace", create_workspace),
        ("generate_pdf", generate_pdf),
        ("build_manifest", build_manifest),
        ("read_artifact_bytes", read_artifact_bytes),
        ("run_bash", run_bash),
        ("cleanup", cleanup),
    ):
        monkeypatch.setattr(RemoteE2BComputer, name, replacement)
    return recorder


def _guard_01_remote_refusal(ctx: Any, workdir: Path) -> None:
    """Distilled from ``test_remote_computer_fail_closed.py::
    test_remote_computer_every_operation_refuses_before_host_access`` — every
    placeholder operation must raise ``SandboxUnavailableError`` before any
    host access, and the host sentinels must record nothing."""
    from agent_v2.computer import SandboxUnavailableError, create_computer

    computer = create_computer("remote_e2b")
    calls: list[tuple[str, Callable[[], Any]]] = [
        ("create_workspace", lambda: computer.create_workspace("run-inj")),
        ("generate_pdf", lambda: computer.generate_pdf("/ws", "content", "plan.pdf")),
        ("build_manifest", lambda: computer.build_manifest("/ws", "plan.pdf", "application/pdf")),
        ("read_artifact_bytes", lambda: computer.read_artifact_bytes("/ws", "plan.pdf")),
        ("run_bash", lambda: computer.run_bash("/ws", "echo hi")),
        ("cleanup", lambda: computer.cleanup("/ws")),
    ]
    leaked: list[str] = []
    for name, call in calls:
        try:
            call()
        except SandboxUnavailableError:
            continue
        leaked.append(name)

    attempts = list(getattr(ctx, "attempts", []) or [])
    if attempts:
        raise AssertionError(
            f"ticket01 guard: host access attempted: {attempts[:3]}"
        )
    if leaked:
        raise AssertionError(
            f"ticket01 guard: operations did not fail closed: {leaked}"
        )


# ---------------------------------------------------------------------------
# ticket 02 — never-searched factual text must be gated
# ---------------------------------------------------------------------------


def _inject_02_ungated_evidence_gate(
    monkeypatch: pytest.MonkeyPatch, workdir: Path
) -> None:
    """Pre-ticket-02 gate: only the fabrication signature (search attempted,
    no chunks) was blocked; a turn that NEVER searched let arbitrary factual
    text through to the user."""
    module = importlib.import_module("agent_v2.run_orchestration")

    def old_gate(
        model_text: str,
        audit_entries: list[dict[str, Any]],
        *,
        req: Any,
        run_id: str | None = None,
    ) -> str:
        searched = any(
            entry.get("tool") == "hiking_knowledge_search" for entry in audit_entries
        )
        if searched and not module.has_hiking_knowledge_evidence(
            audit_entries, run_id=run_id
        ):
            return "资料不足，无法给出可靠回答。"
        return model_text

    monkeypatch.setattr(module, "apply_chat_turn_evidence_gate", old_gate)


def _guard_02_no_ungrounded_free_text(ctx: Any, workdir: Path) -> None:
    """Distilled from ``test_chat_evidence_gate_contract.py::
    test_never_searched_factual_output_is_blocked``."""
    from agent_v2.run_orchestration import ChatFlowRequest, RunOrchestrator

    async def fake_kernel(**kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
        return (UNSUPPORTED_FACT, [])

    orch = RunOrchestrator(
        settings=_settings(),
        runs_repo=_repo(workdir),
        memory=_FakeMemory(),
        kernel_turn=fake_kernel,
    )
    req = ChatFlowRequest(
        message="这段路线爬升多少",
        request_id="req-inj-gate",
        profile_id="chat",
        conversation_id="conv-inj-gate",
    )

    async def run() -> list[Any]:
        return [item async for item in orch.stream_chat(req, subject_id="sub-inj")]

    items = asyncio.run(run())
    text = next(e.payload["content"] for e in items if e.type == "text")
    if UNSUPPORTED_FACT in text or "1800" in text:
        raise AssertionError(
            f"ticket02 guard: ungrounded free text passed through: {text!r}"
        )
    if "资料不足" not in text:
        raise AssertionError(
            f"ticket02 guard: insufficiency template missing: {text!r}"
        )


# ---------------------------------------------------------------------------
# ticket 04 — real PDF bytes + illegal artifact rejection
# ---------------------------------------------------------------------------


def _inject_04_text_stub_pdf_renderer(
    monkeypatch: pytest.MonkeyPatch, workdir: Path
) -> None:
    """Pre-ticket-04 rendering: the "PDF" was the draft TEXT with a .pdf
    suffix."""
    rendering = importlib.import_module("agent_v2.pdf_rendering")

    monkeypatch.setattr(
        rendering,
        "render_markdown_pdf_bytes",
        lambda markdown, **kwargs: markdown.encode("utf-8"),
    )


def _inject_04_permissive_pdf_publish(
    monkeypatch: pytest.MonkeyPatch, workdir: Path
) -> None:
    """Pre-ticket-04 publish path: no PDF content verification at all — the
    artifact was published after the manifest checksum alone."""
    artifacts = importlib.import_module("agent_v2.artifacts")

    monkeypatch.setattr(artifacts, "validate_pdf_bytes", lambda data: None)


def _guard_04a_real_pdf_bytes(ctx: Any, workdir: Path) -> None:
    """Distilled from ``test_pdf_bytes_contract.py::
    test_render_produces_real_parseable_pdf_with_mixed_text``."""
    from pypdf import PdfReader

    from agent_v2.artifacts import validate_pdf_bytes
    from agent_v2.pdf_rendering import render_markdown_pdf_bytes

    data = render_markdown_pdf_bytes(_DRAFT)
    if not data.startswith(b"%PDF"):
        raise AssertionError(
            f"ticket04 guard: rendered artifact is not a real PDF "
            f"(head={data[:16]!r})"
        )
    validate_pdf_bytes(data)  # shared content contract accepts the render
    text = "\n".join(
        (page.extract_text() or "") for page in PdfReader(BytesIO(data)).pages
    )
    if "杭州两日徒步计划" not in text or "Café" not in text:
        raise AssertionError(
            f"ticket04 guard: rendered PDF lost its content: {text[:80]!r}"
        )


def _guard_04b_publish_rejection(ctx: Any, workdir: Path) -> None:
    """Distilled from ``test_pdf_bytes_contract.py::
    test_export_rejects_text_fake_pdf``."""
    from agent_v2.artifacts import ArtifactExportError, export_artifact_to_host

    repo = _repo(workdir)
    run_id, _ = repo.create_run_idempotent(
        request_id="req-inj-pdf",
        conversation_id="conv-inj-pdf",
        profile_id="artifact",
        user_message="规划杭州两日徒步并出 PDF 计划",
    )
    try:
        result = export_artifact_to_host(
            computer=_FakeArtifactComputer(_TEXT_FAKE_PDF),
            workspace_path=f"/ws/{run_id}",
            relative_path="hiking-plan.pdf",
            declared_mime="application/pdf",
            storage_dir=str(workdir / "storage"),
            runs_repo=repo,
            run_id=run_id,
            conversation_id="conv-inj-pdf",
        )
    except ArtifactExportError:
        return  # rejected — the guard's expectation
    raise AssertionError(
        f"ticket04 guard: text-fake PDF was published as an artifact: {result}"
    )


# ---------------------------------------------------------------------------
# ticket 05 — bounded, complete-turn, completed-run history window
# ---------------------------------------------------------------------------


def _inject_05_raw_history_dump(
    monkeypatch: pytest.MonkeyPatch, workdir: Path
) -> None:
    """Pre-ticket-05 store read: a naive raw dump — every stored message,
    un-trimmed, unfinished runs included."""
    from agent_v2.runs import RunRepository

    def old_get_recent_completed_history(
        self: Any,
        *,
        conversation_id: str,
        current_run_id: str | None = None,
        max_messages: int = 0,
        max_chars: int = 0,
    ) -> list[dict[str, str]]:
        page = self.checkpoints.get_messages_newest_first(
            conversation_id=conversation_id, limit=10_000, offset=0
        )
        return [
            {"role": message.role, "content": message.content}
            for message in reversed(page)
            if message.role in ("user", "assistant")
        ]

    monkeypatch.setattr(
        RunRepository, "get_recent_completed_history", old_get_recent_completed_history
    )


def _guard_05_bounded_history(ctx: Any, workdir: Path) -> None:
    """Distilled from ``test_conversation_context_contract.py::
    test_history_window_keeps_at_most_twenty_messages_by_whole_turns`` plus
    the running-run exclusion pinned by ``::test_history_excludes_running_
    failed_and_cancelled_turns``."""
    repo = _repo(workdir)
    for index in range(12):
        _seed_turn(
            repo,
            request_id=f"req-inj-win-{index}",
            user_msg=f"约束{index}",
            assistant_msg=f"回答{index}",
            completed=True,
        )
    _seed_turn(
        repo,
        request_id="req-inj-running",
        user_msg="进行中的新请求",
        assistant_msg="进行中的回答",
        completed=False,
    )

    history = repo.get_recent_completed_history(conversation_id=CONVERSATION)

    if len(history) != 20:
        raise AssertionError(
            f"ticket05 guard: history window is not bounded to 20 messages "
            f"(got {len(history)})"
        )
    if history[0] != {"role": "user", "content": "约束2"} or history[1] != {
        "role": "assistant",
        "content": "回答2",
    }:
        raise AssertionError(
            f"ticket05 guard: window does not keep the newest whole turns: "
            f"{history[:2]!r}"
        )
    if history[-1] != {"role": "assistant", "content": "回答11"}:
        raise AssertionError(f"ticket05 guard: newest turn lost: {history[-1]!r}")
    contents = [entry["content"] for entry in history]
    if "进行中的新请求" in contents or "进行中的回答" in contents:
        raise AssertionError("ticket05 guard: an unfinished run leaked into history")
    if "约束0" in contents or "回答1" in contents:
        raise AssertionError("ticket05 guard: dropped old turns reappeared")


# ---------------------------------------------------------------------------
# the case registry: injections vs the fixed-code control
# ---------------------------------------------------------------------------

_CASES: dict[str, tuple[Callable[[pytest.MonkeyPatch, Path], Any], Callable[[Any, Path], None]]] = {
    "ticket01_remote_host_stub": (_inject_01_remote_host_stub, _guard_01_remote_refusal),
    "ticket02_ungated_evidence_gate": (
        _inject_02_ungated_evidence_gate,
        _guard_02_no_ungrounded_free_text,
    ),
    "ticket04_text_stub_pdf_renderer": (
        _inject_04_text_stub_pdf_renderer,
        _guard_04a_real_pdf_bytes,
    ),
    "ticket04_permissive_pdf_publish": (
        _inject_04_permissive_pdf_publish,
        _guard_04b_publish_rejection,
    ),
    "ticket05_raw_history_dump": (_inject_05_raw_history_dump, _guard_05_bounded_history),
}


def _fixed_01_control(
    monkeypatch: pytest.MonkeyPatch, workdir: Path
) -> _HostAttemptRecorder:
    """Fixed-code control for ticket 01: keep the host sentinels armed but
    inject NO defect — the fixed code must still touch nothing."""
    return _HostAttemptRecorder(monkeypatch)


def _no_injection(monkeypatch: pytest.MonkeyPatch, workdir: Path) -> None:
    return None


_FIXED_CASES: dict[str, tuple[Callable[[pytest.MonkeyPatch, Path], Any], Callable[[Any, Path], None]]] = {
    "ticket01_remote_host_stub": (_fixed_01_control, _guard_01_remote_refusal),
    "ticket02_ungated_evidence_gate": (_no_injection, _guard_02_no_ungrounded_free_text),
    "ticket04_text_stub_pdf_renderer": (_no_injection, _guard_04a_real_pdf_bytes),
    "ticket04_permissive_pdf_publish": (_no_injection, _guard_04b_publish_rejection),
    "ticket05_raw_history_dump": (_no_injection, _guard_05_bounded_history),
}


def inject_case(name: str, monkeypatch: pytest.MonkeyPatch, workdir: Path) -> Any:
    inject, _guard = _CASES[name]
    return inject(monkeypatch, workdir)


def run_injected_case(
    name: str, monkeypatch: pytest.MonkeyPatch, workdir: Path
) -> None:
    """Inject the old defect, then run the key regression guard.

    On a healthy product the guard CATCHES the injected defect and raises
    ``AssertionError`` (message starts with the ticket marker) — that is the
    evidence this contract asserts. If the guard ever stops catching it,
    this function returns quietly and the test FAILS.
    """
    inject, guard = _CASES[name]
    ctx = inject(monkeypatch, workdir / f"{name}-injected")
    guard(ctx, workdir / f"{name}-injected")


def run_fixed_case(name: str, monkeypatch: pytest.MonkeyPatch, workdir: Path) -> None:
    fixed_ctx, guard = _FIXED_CASES[name]
    ctx = fixed_ctx(monkeypatch, workdir / f"{name}-fixed")
    guard(ctx, workdir / f"{name}-fixed")


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_CASES))
def test_injected_old_defect_makes_the_key_regression_fail(
    name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(AssertionError, match="guard") as caught:
        run_injected_case(name, monkeypatch, tmp_path)
    assert "ticket0" in str(caught.value)


@pytest.mark.parametrize("name", sorted(_CASES))
def test_the_same_guard_passes_on_the_fixed_code(
    name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_fixed_case(name, monkeypatch, tmp_path)  # must not raise


def test_ticket01_injection_only_records_attempts_never_executes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctx = inject_case("ticket01_remote_host_stub", monkeypatch, tmp_path)
    with pytest.raises(AssertionError, match="ticket01 guard"):
        _guard_01_remote_refusal(ctx, tmp_path / "ticket01-recorder")
    assert any("mkdtemp" in attempt for attempt in ctx.attempts)
    assert any(
        "subprocess.run" in attempt and "bash" in attempt for attempt in ctx.attempts
    )
    # The sentinel returned an inert path: no host temp dir was created.
    assert not (Path(tempfile.gettempdir()) / "injected-never-created").exists()


def test_injected_defects_also_fail_the_real_contract_suites(tmp_path: Path) -> None:
    """The injections are not only caught by the distilled guards: the REAL
    contract test functions fail under them too (no forked expectations)."""
    import test_chat_evidence_gate_contract as gate_suite
    import test_conversation_context_contract as context_suite
    import test_pdf_bytes_contract as pdf_suite
    from agent_v2.artifacts import ArtifactExportError

    with pytest.MonkeyPatch.context() as mp:
        _inject_02_ungated_evidence_gate(mp, tmp_path)
        with pytest.raises(AssertionError, match="1800"):
            gate_suite.test_never_searched_factual_output_is_blocked(
                tmp_path / "real-gate"
            )

    with pytest.MonkeyPatch.context() as mp:
        # The renderer seam is patched where the PRODUCT resolves it
        # (computer generates through a call-time local import); the real
        # contract test then fails on the injected text stub.
        _inject_04_text_stub_pdf_renderer(mp, tmp_path)
        with pytest.raises((AssertionError, ArtifactExportError), match="PDF"):
            pdf_suite.test_local_native_generate_pdf_writes_real_pdf_bytes(
                tmp_path / "real-render"
            )

    with pytest.MonkeyPatch.context() as mp:
        _inject_04_permissive_pdf_publish(mp, tmp_path)
        with pytest.raises(AssertionError, match="is None"):
            pdf_suite.test_export_rejects_text_fake_pdf(tmp_path / "real-pdf")

    with pytest.MonkeyPatch.context() as mp:
        _inject_05_raw_history_dump(mp, tmp_path)
        with pytest.raises(AssertionError, match="== 20"):
            context_suite.test_history_window_keeps_at_most_twenty_messages_by_whole_turns(
                tmp_path / "real-context"
            )


_JOB_FILE_TEMPLATE = '''"""Generated by ai-service/test_regression_injection_contract.py (ticket 07).

Scenario: the validation job runs the key regressions against a product with
ONE KNOWN OLD DEFECT INJECTED per test (safe sentinel doubles — nothing is
executed). A healthy regression suite FAILS here: every guard catches its
defect. A nonzero pytest exit is therefore the evidence that the guards are
REAL behavioral gates — while `compileall` stays green on this same file
(syntax compilation cannot substitute for behavior tests).
"""

import sys

sys.path.insert(0, {ai_service!r})

import pytest

from test_regression_injection_contract import run_injected_case

CASES = {cases!r}


@pytest.mark.parametrize("case", CASES)
def test_key_regression_catches_the_injected_old_defect(case, monkeypatch, tmp_path):
    # The guard's own AssertionError deliberately propagates: this test IS
    # the "job" the injected defect must turn red.
    run_injected_case(case, monkeypatch, tmp_path)
'''


def test_injected_defects_fail_a_real_pytest_job_while_syntax_stays_green(
    tmp_path: Path,
) -> None:
    """AC4: the KEY regression AND the job really fail on the injected defect.

    A real pytest subprocess runs the injected-defect scenario exactly like
    the CI regression step would. ``compileall`` on the same file stays
    green — the defect is behavioral, so syntax compilation cannot catch it.
    """
    generated = tmp_path / "test_injected_defect_job.py"
    generated.write_text(
        _JOB_FILE_TEMPLATE.format(
            ai_service=str(AI_SERVICE_DIR), cases=sorted(_CASES)
        ),
        encoding="utf-8",
    )

    # 1) Syntax compilation: GREEN (the old CI-only step would pass).
    compile_proc = subprocess.run(
        [sys.executable, "-B", "-m", "compileall", "-q", str(tmp_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert compile_proc.returncode == 0, compile_proc.stdout + compile_proc.stderr

    # 2) The REAL behavior job: RED — every guard catches its injected defect.
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [
            sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
            str(generated),
        ],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, (
        "injected defect did NOT fail the job — the guards are not gating:\n"
        + output
    )
    assert f"{len(_CASES)} failed" in output, output
    for marker in ("ticket01 guard:", "ticket02 guard:", "ticket04 guard:", "ticket05 guard:"):
        assert marker in output, f"{marker} missing from job output:\n{output}"
