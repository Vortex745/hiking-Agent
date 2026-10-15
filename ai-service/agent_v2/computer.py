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

Ticket 01 (harness-p0-p1): ``RemoteE2BComputer`` is NOT a stub that mimics
the E2B lifecycle on the host. No live E2B client is wired yet, so the
placeholder now FAILS CLOSED: every operation raises
:class:`SandboxUnavailableError` before any host workspace, process, or
file is touched. There is no automatic fallback to
:class:`LocalNativeComputer` — a run selecting the remote sandbox ends in
a terminal ``sandbox_unavailable`` error, and basic chat plus explicitly
configured trusted ``local_native`` development keep working unchanged.

Real E2B integration (a later ticket) replaces ``RemoteE2BComputer`` with
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agent_v2.artifacts import ArtifactManifest, compute_sha256

logger = logging.getLogger("ai-service.agent_v2.computer")


SandboxType = Literal["local_native", "wsl_localvm", "remote_e2b"]


# ---- Failure modes (injectable for tests + future E2B mapping) ------------


class ComputerError(Exception):
    """Base for all sandbox failures. Fail-closed: never fall back to host."""


class SandboxCreationError(ComputerError):
    """Sandbox could not be created (e.g. E2B API error, disk full)."""


class SandboxUnavailableError(ComputerError):
    """The selected sandbox backend is a placeholder that refuses to run.

    Ticket 01 (harness-p0-p1): ``remote_e2b`` has no live client yet. The
    placeholder raises this error BEFORE any host workspace, process, or
    file operation — it never mimics the E2B lifecycle on the host, and
    callers must never fall back to a host computer. The public mapping
    is the stable error code ``sandbox_unavailable`` with exactly one
    terminal event per stream; nothing is executed, so no effect is
    committed and no artifact is produced.
    """


class SandboxQuotaError(ComputerError):
    """Quota exhausted — no new sandboxes can be provisioned."""


class SandboxTimeoutError(ComputerError):
    """A command inside the sandbox timed out."""


class SandboxGenerationError(ComputerError):
    """Artifact generation inside the sandbox failed (Ticket 04).

    Raised when the trusted-local sandbox cannot produce a real artifact
    (e.g. blank body, missing rendering dependency). The caller maps it to
    a terminal failure — no pseudo artifact is ever published.
    """


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
        # Ticket 04: real PDF bytes via the shared generic renderer. A
        # blank body or a missing rendering dependency raises
        # PDFRenderingError → mapped by the caller to a terminal failure;
        # a text file with a .pdf suffix is never produced again.
        from agent_v2.pdf_rendering import (
            PDFRenderingError,
            render_markdown_pdf_bytes,
        )

        safe_name = Path(filename).name  # strip any path components
        if not safe_name.endswith(".pdf"):
            safe_name = f"{safe_name}.pdf"
        try:
            data = render_markdown_pdf_bytes(content, title=Path(safe_name).stem)
        except PDFRenderingError as exc:
            raise SandboxGenerationError(f"PDF 渲染失败：{exc}") from exc
        full = Path(workspace_path) / safe_name
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


# ---- RemoteE2B (production; placeholder until a live client lands) ------------

_REFUSAL_MESSAGE = (
    "远程沙箱（remote_e2b）尚未接入：已拒绝执行。"
    "未创建宿主工作区、未启动任何进程、未写入任何文件；"
    "基础聊天与明确配置的受信 local_native 不受影响。"
)


class RemoteE2BComputer(Computer):
    """Production sandbox placeholder that FAILS CLOSED (Ticket 01).

    No live E2B client is wired. The previous stub mimicked the E2B
    lifecycle on the HOST (a temp directory plus ``bash -c``), which meant
    model-written commands executed on the host machine whenever
    ``remote_e2b`` — the production default — was selected. That behavior
    is removed: every operation now raises :class:`SandboxUnavailableError`
    BEFORE any host access, and no automatic fallback to
    :class:`LocalNativeComputer` exists.

    When the real E2B integration lands, it replaces this class body with
    the SDK client; the interface, the fail-closed taxonomy, and the
    ``sandbox_unavailable`` public mapping stay the same.

    ``failure_mode`` is still accepted so the factory signature and test
    injection points stay compatible; it no longer changes any behavior
    because every operation refuses regardless.
    """

    sandbox_type: SandboxType = "remote_e2b"

    def __init__(self, failure_mode: str | None = None) -> None:
        self._failure_mode = failure_mode

    def create_workspace(self, run_id: str) -> str:
        raise SandboxUnavailableError(_REFUSAL_MESSAGE)

    def generate_pdf(
        self,
        workspace_path: str,
        content: str,
        filename: str,
    ) -> str:
        raise SandboxUnavailableError(_REFUSAL_MESSAGE)

    def build_manifest(
        self,
        workspace_path: str,
        relative_path: str,
        declared_mime: str,
    ) -> ArtifactManifest:
        raise SandboxUnavailableError(_REFUSAL_MESSAGE)

    def read_artifact_bytes(
        self,
        workspace_path: str,
        relative_path: str,
    ) -> bytes:
        raise SandboxUnavailableError(_REFUSAL_MESSAGE)

    def run_bash(
        self,
        workspace_path: str,
        command: str,
        *,
        timeout_seconds: float = 30.0,
        max_output_bytes: int = 1_048_576,
    ) -> "BashResult":
        raise SandboxUnavailableError(_REFUSAL_MESSAGE)

    def cleanup(self, workspace_path: str) -> ResourceUsage:
        raise SandboxUnavailableError(_REFUSAL_MESSAGE)


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
