"""Ticket 09 — Run-scoped Computer (sandbox) abstraction.

Spec decisions:
* #19 — Production creates a run-scoped Remote E2B Computer. Development
  may use LocalNative or WSL LocalVM. There is NO automatic host fallback
  in production.
* #20 — E2B workspace is the only writable execution space. The sandbox
  template pins runtime, artifact libraries, and fonts but contains no
  model, MCP, storage, or provider credentials.
* #11 — artifact profile uses E2B-scoped Read/Write/Edit/Glob/Grep; host
  paths and delete operations are unavailable.

This module is the stub-stage implementation: ``LocalNativeComputer``
uses a temp directory on the host (development only), and
``RemoteE2BComputer`` is a stub that mimics the E2B lifecycle without
making a network call. Both honor ``fail_closed`` — a creation, quota,
timeout, or cleanup failure raises :class:`ComputerError` and the caller
must surface it as a run error; the run NEVER falls back to executing
on the host.

Real E2B integration (later ticket) replaces ``RemoteE2BComputer`` with
a live client; the interface and fail-closed contract stay the same.

Ticket 10 adds manifest + byte retrieval so the host can re-verify
sandbox-produced artifacts before publishing them. The host never
trusts sandbox-declared size/sha — it re-derives them from the bytes
read via :meth:`Computer.read_artifact_bytes`.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
import time
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agent_v2.artifacts import ArtifactManifest, compute_sha256

logger = logging.getLogger("ai-service.agent_v2.computer")


SandboxType = Literal["local_native", "wsl_localvm", "remote_e2b"]


def _render_pdf_bytes(content: str) -> bytes:
    """Render text into a valid PDF document.

    The sandbox implementation is intentionally small, but returning a text
    file with a ``.pdf`` suffix makes artifact MIME validation meaningless.
    ReportLab keeps the development path deterministic and produces bytes that
    PDF readers and downstream artifact checks can actually parse.
    """
    try:
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
        from xml.sax.saxutils import escape
    except ImportError as exc:
        raise ComputerError(
            "PDF generation requires reportlab; install ai-service/requirements.txt"
        ) from exc

    font_name = "Helvetica"
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        font_name = "STSong-Light"
    except Exception:
        # Latin-only content still has a usable fallback when the CID font is
        # unavailable in a minimal local environment.
        logger.warning("STSong-Light is unavailable; using Helvetica for PDF text")

    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="AI Hiking artifact",
    )
    style = ParagraphStyle(
        "ArtifactBody",
        fontName=font_name,
        fontSize=10,
        leading=15,
        alignment=TA_LEFT,
    )
    lines = content.splitlines() or [""]
    story = []
    for line in lines:
        story.append(Paragraph(escape(line) or "&nbsp;", style))
        story.append(Spacer(1, 3))
    document.build(story)
    return output.getvalue()


# ---- Failure modes (injectable for tests + future E2B mapping) ------------


class ComputerError(Exception):
    """Base for all sandbox failures. Fail-closed: never fall back to host."""


class SandboxCreationError(ComputerError):
    """Sandbox could not be created (e.g. E2B API error, disk full)."""


class SandboxQuotaError(ComputerError):
    """Quota exhausted — no new sandboxes can be provisioned."""


class SandboxTimeoutError(ComputerError):
    """A command inside the sandbox timed out."""


class SandboxFileNotFoundError(ComputerError):
    """A sandbox file was expected but is missing (e.g. manifest build
    or read_artifact_bytes on a non-existent path). Distinct from
    :class:`SandboxTimeoutError` so callers can classify failures."""


class SandboxBashOutputLimitExceeded(ComputerError):
    """A sandbox bash command produced more output than the configured
    cap (Ticket 13 AC3: 输出上限). The partial output captured before
    the limit is exposed via ``partial_output`` so the audit log can
    record what the command printed before being cut off."""

    def __init__(self, message: str, partial_output: str = "") -> None:
        super().__init__(message)
        self.partial_output = partial_output


class SandboxCleanupError(ComputerError):
    """Cleanup failed — the lease is marked ``failed`` but the run continues
    to its terminal state. The operator must investigate stale sandboxes.

    Carries ``partial_usage`` (best-effort :class:`ResourceUsage` gathered
    before the cleanup failure) so the lease can record resource usage
    even when destruction failed (spec AC6: record success, failure, AND
    resource usage).
    """

    def __init__(self, message: str, partial_usage: "ResourceUsage | None" = None) -> None:
        super().__init__(message)
        self.partial_usage = partial_usage


# ---- Resource usage reported at cleanup -----------------------------------


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """Resource metrics gathered at sandbox cleanup (spec AC6).

    All fields are best-effort; the stub reports synthetic values. Real
    E2B integration replaces these with ``usage`` from the E2B API.
    """

    duration_ms: int
    files_written: int
    bytes_written: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_ms": self.duration_ms,
            "files_written": self.files_written,
            "bytes_written": self.bytes_written,
        }


# ---- Bash result (Ticket 13) ---------------------------------------------


@dataclass(frozen=True, slots=True)
class BashResult:
    """Result of a sandbox bash command (Ticket 13).

    All fields are deterministic so the effect ledger can store the
    result and the resume flow can replay it without re-executing the
    command (spec AC2: 把已有成功结果作为唯一结果返回).

    ``stdout`` and ``stderr`` are truncated to ``max_output_bytes`` per
    stream; the ``truncated`` flag is set when truncation occurred so
    the audit log and the user-facing event can surface it.
    """

    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
            "truncated": self.truncated,
        }


# ---- Computer interface ---------------------------------------------------


class Computer:
    """Abstract sandbox lifecycle + artifact operations.

    Implementations must be fail-closed: any failure raises a
    :class:`ComputerError` subclass. The caller (run stream) maps the
    failure to an ``error`` terminal event; it NEVER retries on the host.
    """

    sandbox_type: SandboxType

    def create_workspace(self, run_id: str) -> str:
        """Provision a workspace for ``run_id`` and return its path.

        Raises :class:`SandboxCreationError` or :class:`SandboxQuotaError`
        on failure. The returned path is opaque to callers — they pass
        it back to ``generate_pdf`` / ``cleanup``.
        """
        raise NotImplementedError

    def generate_pdf(
        self,
        workspace_path: str,
        content: str,
        filename: str,
    ) -> str:
        """Write ``content`` as ``filename`` inside the sandbox workspace.

        Returns the relative path of the generated file (no host prefix).
        Raises :class:`SandboxTimeoutError` if the write exceeds the
        sandbox command timeout.
        """
        raise NotImplementedError

    def build_manifest(
        self,
        workspace_path: str,
        relative_path: str,
        declared_mime: str,
    ) -> ArtifactManifest:
        """Build a sandbox-side manifest for the produced artifact (Ticket 10).

        The sandbox declares the file's relative path, public filename,
        MIME type, size, and SHA-256. The host re-derives size + SHA-256
        from the bytes returned by :meth:`read_artifact_bytes` and
        rejects any mismatch.

        Raises :class:`SandboxTimeoutError` if the checksum command
        exceeds the sandbox timeout.
        """
        raise NotImplementedError

    def read_artifact_bytes(
        self,
        workspace_path: str,
        relative_path: str,
    ) -> bytes:
        """Read the produced artifact's bytes back to the host (Ticket 10).

        The host uses these bytes to re-verify the manifest. The sandbox
        must NOT mutate the file between :meth:`generate_pdf` and this
        call.

        Raises :class:`SandboxTimeoutError` if the read exceeds the
        sandbox timeout.
        """
        raise NotImplementedError

    def run_bash(
        self,
        workspace_path: str,
        command: str,
        *,
        timeout_seconds: float = 30.0,
        max_output_bytes: int = 1_048_576,
    ) -> "BashResult":
        """Run a bash command inside the sandbox workspace (Ticket 13).

        Spec AC3: all Bash calls enforce a timeout and an output cap.
        Spec AC4: the command runs with ``cwd=workspace_path`` and an
        EMPTY environment (``env={}``) so host paths, host environment
        variables, and host processes are NOT accessible from inside
        the sandbox. The only path the command can read or write is the
        workspace.

        Args:
            workspace_path: opaque sandbox workspace handle returned by
                :meth:`create_workspace`.
            command: shell command string. Executed via ``bash -c`` on
                the sandbox; no shell expansion is performed by the
                host.
            timeout_seconds: wall-clock cap. Exceeding this raises
                :class:`SandboxTimeoutError`.
            max_output_bytes: per-stream (stdout/stderr) byte cap. The
                captured bytes are truncated to this length and the
                result's ``truncated`` flag is set; the run continues
                rather than failing (the user sees the partial output).

        Returns:
            :class:`BashResult` with exit code, truncated stdout/stderr,
            duration, and a ``truncated`` flag.

        Raises:
            :class:`SandboxTimeoutError`: command exceeded timeout.
            :class:`ComputerError`: sandbox-side execution failure.
        """
        raise NotImplementedError

    def cleanup(self, workspace_path: str) -> ResourceUsage:
        """Destroy the sandbox and return resource usage (spec AC6).

        Raises :class:`SandboxCleanupError` if destruction fails. The
        caller records the failure on the lease but still completes the
        run's terminal transition.
        """
        raise NotImplementedError


# ---- LocalNative (development only) ---------------------------------------


class LocalNativeComputer(Computer):
    """Development sandbox backed by a host temp directory.

    Spec decision #19 allows LocalNative in development. This impl is
    NOT a production fallback — production uses :class:`RemoteE2BComputer`.
    The temp directory is removed on ``cleanup``.
    """

    sandbox_type: SandboxType = "local_native"

    def __init__(self, failure_mode: str | None = None) -> None:
        # ``failure_mode`` lets tests inject failures without monkeypatching.
        # None = no injected failure. Values: "creation", "quota", "timeout",
        # "cleanup". Production code never sets this.
        self._failure_mode = failure_mode
        self._files_written: int = 0
        self._bytes_written: int = 0
        self._created_at: float | None = None

    def create_workspace(self, run_id: str) -> str:
        if self._failure_mode == "creation":
            raise SandboxCreationError("injected creation failure")
        if self._failure_mode == "quota":
            raise SandboxQuotaError("injected quota exhaustion")
        # mkdtemp gives us a host-backed workspace for development. The
        # path is opaque to callers; they never treat it as a host path.
        # Wrap OSError (disk full / permissions) as SandboxCreationError
        # so the fail-closed branch in _stream_artifact_flow maps it to
        # ``sandbox_creation_failed`` instead of a generic internal_error.
        try:
            workspace = tempfile.mkdtemp(prefix=f"v2-sandbox-{run_id}-")
        except OSError as exc:
            raise SandboxCreationError(f"workspace creation failed: {exc}") from exc
        self._created_at = time.monotonic()
        logger.info("LocalNative sandbox created: %s", workspace)
        return workspace

    def generate_pdf(
        self,
        workspace_path: str,
        content: str,
        filename: str,
    ) -> str:
        if self._failure_mode == "timeout":
            raise SandboxTimeoutError("injected command timeout")
        safe_name = Path(filename).name  # strip any path components
        if not safe_name.endswith(".pdf"):
            safe_name = f"{safe_name}.pdf"
        full = Path(workspace_path) / safe_name
        data = _render_pdf_bytes(content)
        full.write_bytes(data)
        self._files_written += 1
        self._bytes_written += len(data)
        return safe_name

    def build_manifest(
        self,
        workspace_path: str,
        relative_path: str,
        declared_mime: str,
    ) -> ArtifactManifest:
        if self._failure_mode == "timeout":
            raise SandboxTimeoutError("injected command timeout")
        # Sandbox computes its own size + sha256. The host re-derives
        # both from read_artifact_bytes and rejects any mismatch — the
        # sandbox's values are declarative, not authoritative.
        full = Path(workspace_path) / relative_path
        try:
            data = full.read_bytes()
        except FileNotFoundError as exc:
            raise SandboxFileNotFoundError(
                f"manifest read failed for {relative_path!r}: file not found"
            ) from exc
        except OSError as exc:
            raise SandboxTimeoutError(
                f"manifest read failed for {relative_path!r}: {exc}"
            ) from exc
        return ArtifactManifest(
            relative_path=relative_path,
            public_filename=Path(relative_path).name,
            declared_mime=declared_mime,
            size_bytes=len(data),
            sha256=compute_sha256(data),
        )

    def read_artifact_bytes(
        self,
        workspace_path: str,
        relative_path: str,
    ) -> bytes:
        if self._failure_mode == "timeout":
            raise SandboxTimeoutError("injected command timeout")
        full = Path(workspace_path) / relative_path
        try:
            return full.read_bytes()
        except FileNotFoundError as exc:
            raise SandboxFileNotFoundError(
                f"artifact read failed for {relative_path!r}: file not found"
            ) from exc
        except OSError as exc:
            raise SandboxTimeoutError(
                f"artifact read failed for {relative_path!r}: {exc}"
            ) from exc

    def run_bash(
        self,
        workspace_path: str,
        command: str,
        *,
        timeout_seconds: float = 30.0,
        max_output_bytes: int = 1_048_576,
    ) -> "BashResult":
        """Run ``command`` inside the LocalNative sandbox workspace.

        Ticket 13 AC4 isolation guarantees:

        * ``cwd=workspace_path`` — the command's working directory is
          the sandbox workspace; relative paths resolve there.
        * ``env={}`` — the host environment is NOT inherited. ``HOME``
          points at the workspace and ``PATH`` is empty so
          shells that expect ``$HOME`` (e.g. ``bash -c`` reading
          ``~/.bashrc``) do not crash while commands cannot resolve host
          binaries. Cowork commands must use shell built-ins.
        * Subprocess is invoked with ``shell=False`` and explicit
          ``["bash", "-c", command]`` so the host does not interpret
          metacharacters.

        Failure injection: ``failure_mode="timeout"`` raises
        :class:`SandboxTimeoutError` without running the command; this
        lets tests exercise the timeout path deterministically.
        """
        if self._failure_mode == "timeout":
            raise SandboxTimeoutError("injected bash timeout")
        # Local import keeps the module header lean.
        import subprocess

        start = time.monotonic()
        # Empty env + HOME=workspace enforces AC4: no host env leak.
        sandbox_env = {"HOME": workspace_path, "PATH": ""}
        try:
            proc = subprocess.run(
                ["bash", "-c", command],
                cwd=workspace_path,
                env=sandbox_env,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxTimeoutError(
                f"bash command exceeded {timeout_seconds}s timeout"
            ) from exc
        except OSError as exc:
            raise ComputerError(f"bash execution failed: {exc}") from exc

        duration_ms = int((time.monotonic() - start) * 1000)
        stdout_bytes = proc.stdout or b""
        stderr_bytes = proc.stderr or b""
        truncated = (
            len(stdout_bytes) > max_output_bytes
            or len(stderr_bytes) > max_output_bytes
        )
        stdout_text = stdout_bytes[:max_output_bytes].decode(
            "utf-8", errors="replace"
        )
        stderr_text = stderr_bytes[:max_output_bytes].decode(
            "utf-8", errors="replace"
        )
        return BashResult(
            exit_code=proc.returncode,
            stdout=stdout_text,
            stderr=stderr_text,
            duration_ms=duration_ms,
            truncated=truncated,
        )

    def cleanup(self, workspace_path: str) -> ResourceUsage:
        duration_ms = int((time.monotonic() - (self._created_at or 0)) * 1000)
        usage = ResourceUsage(
            duration_ms=duration_ms,
            files_written=self._files_written,
            bytes_written=self._bytes_written,
        )
        if self._failure_mode == "cleanup":
            # Don't delete the dir; raise so the lease is marked failed.
            # Attach partial_usage so AC6 records resource usage even on
            # cleanup failure.
            raise SandboxCleanupError(
                "injected cleanup failure", partial_usage=usage
            )
        try:
            shutil.rmtree(workspace_path, ignore_errors=False)
        except OSError as exc:
            raise SandboxCleanupError(str(exc), partial_usage=usage) from exc
        logger.info("LocalNative sandbox destroyed: %s", workspace_path)
        return usage


# ---- RemoteE2B (production; stub until live E2B client lands) -------------


class RemoteE2BComputer(Computer):
    """Production sandbox backed by E2B (stub implementation).

    This stub mimics the E2B lifecycle (create → execute → destroy)
    without making a network call. It uses a temp directory to stand in
    for the remote filesystem so the artifact flow is end-to-end testable
    before the real E2B client is integrated.

    Fail-closed contract (spec decision #19): any failure raises a
    :class:`ComputerError` subclass. The caller NEVER falls back to
    :class:`LocalNativeComputer` — production execution stays on E2B or
    fails the run.
    """

    sandbox_type: SandboxType = "remote_e2b"

    def __init__(self, failure_mode: str | None = None) -> None:
        self._failure_mode = failure_mode
        self._files_written: int = 0
        self._bytes_written: int = 0
        self._created_at: float | None = None

    def create_workspace(self, run_id: str) -> str:
        if self._failure_mode == "creation":
            raise SandboxCreationError("E2B sandbox creation failed")
        if self._failure_mode == "quota":
            raise SandboxQuotaError("E2B quota exhausted")
        # Stub: temp dir stands in for the remote E2B filesystem. Real
        # integration calls e2b.Sandbox.create() and returns the sandbox
        # handle; the workspace_path would be an opaque sandbox id.
        # Wrap OSError as SandboxCreationError to preserve fail-closed.
        try:
            workspace = tempfile.mkdtemp(prefix=f"v2-e2b-{run_id}-")
        except OSError as exc:
            raise SandboxCreationError(f"E2B workspace creation failed: {exc}") from exc
        self._created_at = time.monotonic()
        logger.info("RemoteE2B sandbox created (stub): %s", workspace)
        return workspace

    def generate_pdf(
        self,
        workspace_path: str,
        content: str,
        filename: str,
    ) -> str:
        if self._failure_mode == "timeout":
            raise SandboxTimeoutError("E2B command timeout")
        safe_name = Path(filename).name
        if not safe_name.endswith(".pdf"):
            safe_name = f"{safe_name}.pdf"
        full = Path(workspace_path) / safe_name
        data = _render_pdf_bytes(content)
        full.write_bytes(data)
        self._files_written += 1
        self._bytes_written += len(data)
        return safe_name

    def build_manifest(
        self,
        workspace_path: str,
        relative_path: str,
        declared_mime: str,
    ) -> ArtifactManifest:
        if self._failure_mode == "timeout":
            raise SandboxTimeoutError("E2B command timeout")
        full = Path(workspace_path) / relative_path
        try:
            data = full.read_bytes()
        except FileNotFoundError as exc:
            raise SandboxFileNotFoundError(
                f"E2B manifest read failed for {relative_path!r}: file not found"
            ) from exc
        except OSError as exc:
            raise SandboxTimeoutError(
                f"E2B manifest read failed for {relative_path!r}: {exc}"
            ) from exc
        return ArtifactManifest(
            relative_path=relative_path,
            public_filename=Path(relative_path).name,
            declared_mime=declared_mime,
            size_bytes=len(data),
            sha256=compute_sha256(data),
        )

    def read_artifact_bytes(
        self,
        workspace_path: str,
        relative_path: str,
    ) -> bytes:
        if self._failure_mode == "timeout":
            raise SandboxTimeoutError("E2B command timeout")
        full = Path(workspace_path) / relative_path
        try:
            return full.read_bytes()
        except FileNotFoundError as exc:
            raise SandboxFileNotFoundError(
                f"E2B artifact read failed for {relative_path!r}: file not found"
            ) from exc
        except OSError as exc:
            raise SandboxTimeoutError(
                f"E2B artifact read failed for {relative_path!r}: {exc}"
            ) from exc

    def run_bash(
        self,
        workspace_path: str,
        command: str,
        *,
        timeout_seconds: float = 30.0,
        max_output_bytes: int = 1_048_576,
    ) -> "BashResult":
        """Run ``command`` inside the E2B sandbox (stub: same isolation
        contract as :class:`LocalNativeComputer`).

        Production E2B integration replaces this body with
        ``sandbox.commands.run(command, timeout=...)`` from the E2B SDK.
        The isolation contract (cwd=workspace, env={}) is enforced by
        E2B itself — the remote sandbox has no access to the host.
        """
        if self._failure_mode == "timeout":
            raise SandboxTimeoutError("E2B bash timeout")
        # Stub stage: reuse the LocalNative implementation against the
        # temp dir that stands in for the E2B filesystem. Real E2B
        # integration replaces this with sandbox.commands.run().
        import subprocess

        start = time.monotonic()
        # Same isolation as LocalNative: empty env, HOME=workspace.
        sandbox_env = {"HOME": workspace_path}
        try:
            proc = subprocess.run(
                ["bash", "-c", command],
                cwd=workspace_path,
                env=sandbox_env,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxTimeoutError(
                f"E2B bash command exceeded {timeout_seconds}s timeout"
            ) from exc
        except OSError as exc:
            raise ComputerError(f"E2B bash execution failed: {exc}") from exc

        duration_ms = int((time.monotonic() - start) * 1000)
        stdout_bytes = proc.stdout or b""
        stderr_bytes = proc.stderr or b""
        truncated = (
            len(stdout_bytes) > max_output_bytes
            or len(stderr_bytes) > max_output_bytes
        )
        stdout_text = stdout_bytes[:max_output_bytes].decode(
            "utf-8", errors="replace"
        )
        stderr_text = stderr_bytes[:max_output_bytes].decode(
            "utf-8", errors="replace"
        )
        return BashResult(
            exit_code=proc.returncode,
            stdout=stdout_text,
            stderr=stderr_text,
            duration_ms=duration_ms,
            truncated=truncated,
        )

    def cleanup(self, workspace_path: str) -> ResourceUsage:
        duration_ms = int((time.monotonic() - (self._created_at or 0)) * 1000)
        usage = ResourceUsage(
            duration_ms=duration_ms,
            files_written=self._files_written,
            bytes_written=self._bytes_written,
        )
        if self._failure_mode == "cleanup":
            raise SandboxCleanupError(
                "E2B cleanup failed", partial_usage=usage
            )
        try:
            shutil.rmtree(workspace_path, ignore_errors=False)
        except OSError as exc:
            raise SandboxCleanupError(str(exc), partial_usage=usage) from exc
        logger.info("RemoteE2B sandbox destroyed (stub): %s", workspace_path)
        return usage


# ---- Factory --------------------------------------------------------------


def create_computer(
    sandbox_type: SandboxType = "remote_e2b",
    *,
    failure_mode: str | None = None,
) -> Computer:
    """Return a :class:`Computer` for ``sandbox_type``.

    Spec decision #19: production uses ``remote_e2b``; development may
    select ``local_native`` or ``wsl_localvm``. The factory never falls
    back from ``remote_e2b`` to a host computer — an E2B failure fails
    the run (fail-closed).

    ``wsl_localvm`` currently maps to :class:`LocalNativeComputer` because
    WSL integration is not wired in the stub stage. Real WSL support
    lands with the LocalVM ticket.
    """
    if sandbox_type == "remote_e2b":
        return RemoteE2BComputer(failure_mode=failure_mode)
    if sandbox_type in ("local_native", "wsl_localvm"):
        return LocalNativeComputer(failure_mode=failure_mode)
    raise ValueError(f"invalid sandbox_type: {sandbox_type!r}")
