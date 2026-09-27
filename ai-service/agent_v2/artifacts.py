"""Ticket 10 — Artifact manifest + host-side export verification.

The sandbox produces an :class:`ArtifactManifest` describing the file it
wants to export. The host re-verifies every field against the actual
bytes it reads back from the sandbox, then persists the artifact under
an opaque id. The wire never carries sandbox or host paths.

Spec acceptance criteria covered here:

* AC1 — manifest carries relative path, public filename, declared MIME,
  size, SHA-256; the host re-verifies each field.
* AC2 — absolute paths, ``..``, symlink escape, missing file, MIME
  mismatch, checksum mismatch, and files over 25 MB are rejected.
* AC3 — successful export produces an opaque ``artifact_id``; events
  and download responses do not expose sandbox or host paths.
* AC4 — export failure preserves evidence (manifest + failure reason)
  in the artifacts table BEFORE the sandbox is destroyed.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

logger = logging.getLogger("ai-service.agent_v2.artifacts")


# Spec AC2: 25 MiB hard cap. A single artifact larger than this is
# rejected outright — the run fails with a diagnostic and no half-built
# artifact is published.
MAX_ARTIFACT_BYTES = 25 * 1024 * 1024


# Spec AC2: MIME types the host will accept from a sandbox manifest.
# Anything outside this set is rejected before the bytes are read.
_ALLOWED_MIME_TYPES: frozenset[str] = frozenset({
    "application/pdf",
    "text/plain",
    "text/markdown",
    "application/json",
    "text/csv",
    "text/html",
})


# File extension → declared MIME map used when the manifest declares
# ``application/octet-stream`` (sandbox does not know MIME). The host
# re-derives the MIME from the public filename extension and rejects
# the manifest if the derived MIME is not in the allowlist.
_EXTENSION_MIME_MAP: dict[str, str] = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".json": "application/json",
    ".csv": "text/csv",
    ".html": "text/html",
}


class ArtifactExportError(Exception):
    """Manifest validation or host-side export failed (Ticket 10).

    Carries ``manifest`` so the caller can persist the rejected
    manifest as evidence (spec AC4) before destroying the sandbox.
    """

    def __init__(
        self,
        message: str,
        *,
        manifest: "ArtifactManifest | None" = None,
    ) -> None:
        super().__init__(message)
        self.manifest = manifest


class ArtifactEvidencePersistenceError(Exception):
    """Failed to persist a ``failed`` evidence row (spec AC4).

    Raised by ``_persist_failed_artifact`` when the DB insert itself
    fails. Spec AC4 requires the sandbox NOT be destroyed before
    evidence is saved — the caller (``_stream_artifact_flow``) catches
    this exception and skips ``_cleanup_sandbox`` so the sandbox files
    survive for operator investigation, at the cost of a leaked lease.
    The run still terminates with an ``error`` event.
    """


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Manifest produced inside the sandbox, verified by the host.

    All fields are declarative — the sandbox states what it believes
    the file is. The host re-derives size + SHA-256 from the actual
    bytes and rejects any mismatch (spec AC1, AC2).
    """

    relative_path: str
    public_filename: str
    declared_mime: str
    size_bytes: int
    sha256: str


def validate_relative_path(relative_path: str) -> PurePosixPath:
    """Return the parsed path if safe; raise :class:`ArtifactExportError` otherwise.

    Spec AC2 rejects: absolute paths, ``..`` segments, symlink escape,
    backslash separators (Windows path injection), empty paths.
    """
    if not relative_path or not relative_path.strip():
        raise ArtifactExportError("manifest relative_path is empty")

    # Reject backslash outright — sandbox paths are POSIX.
    if "\\" in relative_path:
        raise ArtifactExportError(
            f"manifest relative_path contains backslash: {relative_path!r}"
        )

    # Reject empty segments (e.g. ``a//b``) BEFORE PurePosixPath collapses
    # them — ``PurePosixPath("a//b").parts == ("a", "b")`` so the parts
    # check below would miss the smuggled segment.
    if "//" in relative_path:
        raise ArtifactExportError(
            f"manifest relative_path has empty segment: {relative_path!r}"
        )

    parsed = PurePosixPath(relative_path)

    # Reject absolute paths (leading "/").
    if parsed.is_absolute():
        raise ArtifactExportError(
            f"manifest relative_path is absolute: {relative_path!r}"
        )

    # Reject any ``..`` segment anywhere in the path.
    if ".." in parsed.parts:
        raise ArtifactExportError(
            f"manifest relative_path contains '..': {relative_path!r}"
        )

    # Reject empty segments (e.g. ``a//b``) which can be used to smuggle
    # path components past naive filters.
    if any(part == "" for part in parsed.parts):
        raise ArtifactExportError(
            f"manifest relative_path has empty segment: {relative_path!r}"
        )

    return parsed


def validate_public_filename(public_filename: str) -> str:
    """Return the basename if safe; raise otherwise.

    The public filename is what the download endpoint serves as
    ``Content-Disposition: filename=...``. It must be a bare basename
    (no path separators, no ``..``) and must map to an allowlisted
    MIME via its extension.
    """
    if not public_filename or not public_filename.strip():
        raise ArtifactExportError("manifest public_filename is empty")

    if "\\" in public_filename or "/" in public_filename:
        raise ArtifactExportError(
            f"public_filename contains path separator: {public_filename!r}"
        )

    if public_filename in (".", ".."):
        raise ArtifactExportError(
            f"public_filename is path traversal: {public_filename!r}"
        )

    if public_filename.startswith(".") and public_filename.count(".") == 1:
        # Hidden files like ``.bashrc`` are not artifacts.
        raise ArtifactExportError(
            f"public_filename is hidden: {public_filename!r}"
        )

    return public_filename


def validate_declared_mime(declared_mime: str, public_filename: str) -> str:
    """Return the MIME if it is allowlisted; raise otherwise.

    If the sandbox declares ``application/octet-stream``, the host
    derives the MIME from the public filename extension. The derived
    MIME must be in the allowlist — this prevents the sandbox from
    serving a ``.exe`` with ``application/pdf``.
    """
    if not declared_mime:
        raise ArtifactExportError("manifest declared_mime is empty")

    if declared_mime == "application/octet-stream":
        ext = Path(public_filename).suffix.lower()
        derived = _EXTENSION_MIME_MAP.get(ext)
        if derived is None:
            raise ArtifactExportError(
                f"cannot derive MIME for extension {ext!r} "
                f"(public_filename={public_filename!r})"
            )
        return derived

    if declared_mime not in _ALLOWED_MIME_TYPES:
        raise ArtifactExportError(
            f"declared_mime not in allowlist: {declared_mime!r}"
        )

    # Cross-check: if the public filename has a known extension, the
    # declared MIME must match it. This blocks ``report.pdf`` declaring
    # ``text/html``.
    ext = Path(public_filename).suffix.lower()
    expected = _EXTENSION_MIME_MAP.get(ext)
    if expected is not None and expected != declared_mime:
        raise ArtifactExportError(
            f"declared_mime {declared_mime!r} does not match "
            f"extension {ext!r} (expected {expected!r})"
        )

    return declared_mime


def validate_size_bytes(size_bytes: int) -> int:
    """Return the size if within the 25 MiB cap; raise otherwise."""
    if size_bytes <= 0:
        raise ArtifactExportError(
            f"manifest size_bytes must be positive, got {size_bytes}"
        )
    if size_bytes > MAX_ARTIFACT_BYTES:
        raise ArtifactExportError(
            f"manifest size_bytes {size_bytes} exceeds cap "
            f"{MAX_ARTIFACT_BYTES} (25 MiB)"
        )
    return size_bytes


def compute_sha256(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def validate_manifest(
    manifest: ArtifactManifest,
    actual_bytes: bytes,
) -> None:
    """Re-verify every manifest field against ``actual_bytes`` (spec AC1, AC2).

    Raises :class:`ArtifactExportError` on any mismatch. The caller
    persists the rejected manifest as evidence (spec AC4) before
    destroying the sandbox.

    The host re-derives size + SHA-256 from the bytes it actually read
    — never trusts the sandbox's declared values.
    """
    validate_relative_path(manifest.relative_path)
    validate_public_filename(manifest.public_filename)
    validate_declared_mime(manifest.declared_mime, manifest.public_filename)
    validate_size_bytes(manifest.size_bytes)

    actual_size = len(actual_bytes)
    if actual_size != manifest.size_bytes:
        raise ArtifactExportError(
            f"size mismatch: manifest declares {manifest.size_bytes} bytes, "
            f"actual is {actual_size} bytes"
        )

    actual_sha = compute_sha256(actual_bytes)
    # Constant-time compare to avoid timing oracles on the digest.
    if not _constant_time_eq(actual_sha, manifest.sha256):
        raise ArtifactExportError(
            f"sha256 mismatch: manifest declares {manifest.sha256}, "
            f"actual is {actual_sha}"
        )


def is_symlink_escape(workspace_root: str, resolved_target: str) -> bool:
    """True if ``resolved_target`` escapes ``workspace_root`` via symlink.

    Spec AC2: symlink escape must be rejected. The host resolves the
    real path of the target file and checks it still lives under the
    workspace root. A symlink pointing outside (or to a non-existent
    path) is rejected.
    """
    root_real = os.path.realpath(workspace_root)
    target_real = os.path.realpath(resolved_target)
    # ``os.path.commonpath`` raises ValueError on cross-drive comparisons
    # (Windows). In that case the target is definitely outside the root.
    try:
        common = os.path.commonpath([root_real, target_real])
    except ValueError:
        return True
    return common != root_real


def host_storage_path(
    artifacts_dir: str,
    artifact_id: str,
    public_filename: str,
) -> str:
    """Return the host-side storage path for an exported artifact.

    Layout: ``<artifacts_dir>/<artifact_id>/<public_filename>``. The
    ``artifact_id`` directory prevents filename collisions across runs
    and ensures the stored filename matches the public filename (so
    ``FileResponse`` can serve it with the right ``Content-Disposition``).
    """
    ext = Path(public_filename).suffix.lower()
    if ext not in _EXTENSION_MIME_MAP:
        # Defensive — validate_manifest should have caught this earlier.
        raise ArtifactExportError(
            f"refusing to construct storage path for disallowed extension: {ext!r}"
        )
    return str(Path(artifacts_dir) / artifact_id / public_filename)


def persist_artifact_bytes(
    storage_path: str,
    data: bytes,
) -> None:
    """Write ``data`` to ``storage_path`` atomically.

    Spec AC4: the host must not publish a half-built artifact. We write
    to a ``.tmp`` sibling and rename, so a crash mid-write leaves no
    partial file at the final path.
    """
    target = Path(storage_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, target)
    except Exception:
        # Clean up the tmp file if the rename failed.
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _constant_time_eq(a: str, b: str) -> bool:
    """Constant-time string equality (delegates to ``hmac.compare_digest``)."""
    import hmac

    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


# ---- Host-side verified export (Ticket 41, formerly adapter code) -----------
#
# Spec AC3 + decision: artifacts persist for 7 days after creation. The
# TTL is intentionally short — artifacts are downloadable summaries, not
# permanent user files. Operators may sweep expired rows + files in a
# separate cron; the download endpoint returns 410 once ``expires_at``
# elapses.
ARTIFACT_TTL_DAYS = 7


def _now_iso() -> str:
    """UTC ISO 8601 timestamp with timezone suffix."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _expires_at_iso(created_at: str) -> str:
    """Compute the expiry timestamp ``ARTIFACT_TTL_DAYS`` after ``created_at``.

    Parses ``created_at`` (ISO 8601 with timezone) so the expiry is
    derived from the creation time, not from ``now`` — this keeps the
    TTL stable even if the clock drifts between insert and expiry check.
    """
    from datetime import datetime, timedelta

    created = datetime.fromisoformat(created_at)
    return (created + timedelta(days=ARTIFACT_TTL_DAYS)).isoformat()


def persist_failed_artifact(
    runs_repo,
    run_id: str | None,
    conversation_id: str,
    artifact_id: str,
    *,
    public_filename: str,
    mime_type: str,
    size_bytes: int,
    sha256: str,
    storage_path: str,
    failure_reason: str,
) -> None:
    """Persist a ``failed`` evidence row (spec AC4).

    Called BEFORE the sandbox is destroyed so the manifest + failure
    reason survive cleanup. Spec AC4 requires "sandbox 不会在证据保存前
    被销毁" — therefore persistence failure here MUST propagate so the
    caller can preserve the sandbox for investigation instead of
    destroying evidence.
    """
    if runs_repo is None or run_id is None:
        return
    created_at = _now_iso()
    expires_at = _expires_at_iso(created_at)
    try:
        runs_repo.insert_artifact(
            artifact_id=artifact_id,
            run_id=run_id,
            conversation_id=conversation_id,
            public_filename=public_filename,
            mime_type=mime_type,
            size_bytes=size_bytes,
            sha256=sha256,
            storage_path=storage_path,
            status="failed",
            failure_reason=failure_reason,
            created_at=created_at,
            expires_at=expires_at,
        )
    except Exception:
        logger.exception(
            "artifacts: failed to persist evidence row for artifact_id=%s; "
            "raising to preserve sandbox per AC4",
            artifact_id,
        )
        raise ArtifactEvidencePersistenceError(
            f"failed to persist evidence row for artifact_id={artifact_id}"
        )


def export_artifact_to_host(
    *,
    computer: Any,
    workspace_path: str,
    relative_path: str,
    declared_mime: str,
    storage_dir: str,
    runs_repo=None,
    run_id: str | None,
    conversation_id: str,
) -> dict[str, Any]:
    """Build manifest, verify on host, persist artifact, record in DB.

    Returns a dict with the public metadata for the ``artifact.available``
    event payload: ``artifact_id``, ``public_filename``, ``mime_type``,
    ``size_bytes``, ``sha256``, ``expires_at``. Spec AC3: no sandbox or
    host paths in the returned dict.

    Spec AC1 + AC2: the host re-derives size + SHA-256 from the bytes
    it reads via :meth:`Computer.read_artifact_bytes` and rejects any
    mismatch with the sandbox-declared manifest. Absolute paths, ``..``,
    symlink escape, missing file, MIME mismatch, checksum mismatch, and
    files over 25 MB are rejected.

    Spec AC4: on ANY failure (manifest build, byte read, validation,
    host persist, DB insert) a ``failed`` evidence row is persisted
    BEFORE this function returns/raises. The caller's ``finally`` block
    may then safely destroy the sandbox — the evidence survives.

    Raises :class:`ArtifactExportError` on failure. The caller emits a
    terminal ``error`` event; no half-built artifact is published.
    """
    artifact_id = str(uuid.uuid4())

    # 1. Sandbox builds the manifest (declares size + sha + mime).
    try:
        manifest = computer.build_manifest(
            workspace_path, relative_path, declared_mime
        )
    except ComputerError as exc:
        # Manifest build failed before we have any declarative fields.
        # Persist a failed row with the best-effort fields we can derive
        # from the request (relative_path → public_filename).
        persist_failed_artifact(
            runs_repo, run_id, conversation_id, artifact_id,
            public_filename=Path(relative_path).name,
            mime_type=declared_mime,
            size_bytes=0,
            sha256="",
            storage_path="",
            failure_reason=f"manifest build failed: {exc}",
        )
        raise ArtifactExportError(f"manifest build failed: {exc}") from exc

    # 2. Host reads the bytes back. The host re-derives size + sha from
    # these bytes — the sandbox's declared values are NOT trusted.
    try:
        actual_bytes = computer.read_artifact_bytes(workspace_path, relative_path)
    except ComputerError as exc:
        persist_failed_artifact(
            runs_repo, run_id, conversation_id, artifact_id,
            public_filename=manifest.public_filename,
            mime_type=manifest.declared_mime,
            size_bytes=manifest.size_bytes,
            sha256=manifest.sha256,
            storage_path="",
            failure_reason=f"artifact read failed: {exc}",
        )
        raise ArtifactExportError(
            f"artifact read failed: {exc}", manifest=manifest
        ) from exc

    # 3. Validate manifest against actual bytes (spec AC1, AC2).
    #    validate_manifest re-derives size + sha256 from actual_bytes
    #    and rejects any mismatch. It also rejects path traversal,
    #    symlink escape (via relative_path validation), MIME mismatch,
    #    and >25 MB files.
    try:
        validate_manifest(manifest, actual_bytes)
    except ArtifactExportError:
        # validate_manifest already populated the manifest on the exc
        # via ArtifactExportError.__init__ when raised from validators
        # that have the manifest in scope. Re-raise after evidence row.
        persist_failed_artifact(
            runs_repo, run_id, conversation_id, artifact_id,
            public_filename=manifest.public_filename,
            mime_type=manifest.declared_mime,
            size_bytes=manifest.size_bytes,
            sha256=manifest.sha256,
            storage_path="",
            failure_reason="manifest validation failed",
        )
        raise

    # 4. Symlink escape check (defense-in-depth, spec AC2). The
    #    relative_path validator already rejected ``..`` and absolute
    #    paths; this catches symlinks inside the workspace that point
    #    outside it.
    full_path = os.path.join(workspace_path, relative_path)
    if is_symlink_escape(workspace_path, full_path):
        persist_failed_artifact(
            runs_repo, run_id, conversation_id, artifact_id,
            public_filename=manifest.public_filename,
            mime_type=manifest.declared_mime,
            size_bytes=manifest.size_bytes,
            sha256=manifest.sha256,
            storage_path="",
            failure_reason="symlink escape detected",
        )
        raise ArtifactExportError(
            "symlink escape detected", manifest=manifest
        )

    # 5. Compute host storage path + persist bytes atomically.
    storage_path = host_storage_path(
        storage_dir, artifact_id, manifest.public_filename
    )
    try:
        persist_artifact_bytes(storage_path, actual_bytes)
    except Exception as exc:
        persist_failed_artifact(
            runs_repo, run_id, conversation_id, artifact_id,
            public_filename=manifest.public_filename,
            mime_type=manifest.declared_mime,
            size_bytes=manifest.size_bytes,
            sha256=manifest.sha256,
            storage_path=storage_path,
            failure_reason=f"host persist failed: {exc}",
        )
        raise ArtifactExportError(
            f"host persist failed: {exc}", manifest=manifest
        ) from exc

    # 6. Insert the artifact row (status='available'). The row is the
    #    source of truth for the download endpoint — the file on disk
    #    is the blob, the row carries the public metadata + expiry.
    created_at = _now_iso()
    expires_at = _expires_at_iso(created_at)
    if runs_repo is not None and run_id is not None:
        try:
            runs_repo.insert_artifact(
                artifact_id=artifact_id,
                run_id=run_id,
                conversation_id=conversation_id,
                public_filename=manifest.public_filename,
                mime_type=manifest.declared_mime,
                size_bytes=manifest.size_bytes,
                sha256=manifest.sha256,
                storage_path=storage_path,
                status="available",
                failure_reason=None,
                created_at=created_at,
                expires_at=expires_at,
            )
        except Exception as exc:
            # DB insert failed after the file was written — best-effort
            # delete the orphan so storage does not accumulate blobs
            # with no matching row. The run still fails; the operator
            # checks logs for the orphan cleanup.
            try:
                os.remove(storage_path)
            except OSError:
                logger.warning(
                    "artifacts: orphan artifact file left at %s after DB insert failed",
                    storage_path,
                )
            persist_failed_artifact(
                runs_repo, run_id, conversation_id, artifact_id,
                public_filename=manifest.public_filename,
                mime_type=manifest.declared_mime,
                size_bytes=manifest.size_bytes,
                sha256=manifest.sha256,
                storage_path="",
                failure_reason=f"artifact DB insert failed: {exc}",
            )
            raise ArtifactExportError(
                f"artifact DB insert failed: {exc}", manifest=manifest
            ) from exc

    return {
        "artifact_id": artifact_id,
        "public_filename": manifest.public_filename,
        "mime_type": manifest.declared_mime,
        "size_bytes": manifest.size_bytes,
        "sha256": manifest.sha256,
        "expires_at": expires_at,
    }


__all__ = [
    "ARTIFACT_TTL_DAYS",
    "ArtifactEvidencePersistenceError",
    "ArtifactExportError",
    "ArtifactManifest",
    "MAX_ARTIFACT_BYTES",
    "compute_sha256",
    "export_artifact_to_host",
    "host_storage_path",
    "is_symlink_escape",
    "persist_artifact_bytes",
    "persist_failed_artifact",
    "validate_declared_mime",
    "validate_manifest",
    "validate_public_filename",
    "validate_relative_path",
    "validate_size_bytes",
]
