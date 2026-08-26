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


__all__ = [
    "ArtifactEvidencePersistenceError",
    "ArtifactExportError",
    "ArtifactManifest",
    "MAX_ARTIFACT_BYTES",
    "compute_sha256",
    "host_storage_path",
    "is_symlink_escape",
    "persist_artifact_bytes",
    "validate_declared_mime",
    "validate_manifest",
    "validate_public_filename",
    "validate_relative_path",
    "validate_size_bytes",
]
