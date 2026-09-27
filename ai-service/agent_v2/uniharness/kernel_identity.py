"""Kernel identity resolution for the vendored UniHarness fork.

Ticket 25 requires that Agent v2 can prove the actually-installed UniHarness
distribution is the project-controlled vendor tree pinned at an immutable
commit, and fail closed when it is missing or incompatible. This module is
the single place that answers "what kernel is really loaded?" — startup and
health read from here; they must never trust configuration alone.
"""

from __future__ import annotations

import importlib.metadata
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("ai-service.agent_v2.uniharness")

# The vendored upstream revision every deployment descends from. Recorded
# here as code (not only .env) so the fail-closed comparison has a built-in
# baseline: a compatible kernel's HEAD is EXACTLY ``settings.agent_v2_uniharness_pin``
# AND this upstream revision is an ancestor of that HEAD (proving the build
# comes from our controlled delta branch on top of upstream).
VENDORED_UPSTREAM_REVISION = "757bf3d60f763706aae60b040d775e8563b9c577"
FORK_PROVENANCE_DOC = "third_party/uniharness-provenance.md"


@dataclass(frozen=True, slots=True)
class KernelIdentity:
    """Resolved identity of the installed UniHarness distribution.

    ``compatible`` is True only when:

    * the ``uniharness`` package is importable/installed, AND
    * its source tree is the vendor checkout whose ``git rev-parse HEAD``
      equals the expected pin commit.

    Any other state is incompatible; callers must treat it as fatal when
    Agent v2 is enabled (fail closed), never as a warning.
    """

    distribution_version: str | None
    package_path: str | None
    fork_commit: str | None
    expected_pin: str
    compatible: bool
    detail: str


def _resolve_fork_commit(package_path: Path) -> str | None:
    """Return ``git rev-parse HEAD`` of the source tree owning the package."""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "rev-parse", "HEAD"],
            cwd=str(package_path),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("git rev-parse failed under %s: %s", package_path, exc)
        return None
    if completed.returncode != 0:
        logger.warning(
            "git rev-parse exited %s under %s: %s",
            completed.returncode,
            package_path,
            completed.stderr.strip()[:200],
        )
        return None
    return completed.stdout.strip() or None


def _pin_is_ancestor_of_head(package_path: Path, revision: str) -> bool | None:
    """Return True when ``revision`` is an ancestor of the source tree HEAD.

    Returns None when git cannot answer (no repo, timeout); callers treat
    None as incompatible — unknown beats guessed.
    """
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "merge-base", "--is-ancestor", revision, "HEAD"],
            cwd=str(package_path),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("git merge-base failed under %s: %s", package_path, exc)
        return None
    if completed.returncode == 0:
        return True
    # rc==1 means "not an ancestor" (a legitimate git answer, not an error).
    if completed.returncode == 1:
        return False
    logger.warning(
        "git merge-base exited %s under %s: %s",
        completed.returncode,
        package_path,
        completed.stderr.strip()[:200],
    )
    return None


def resolve_kernel_identity(expected_pin: str) -> KernelIdentity:
    """Resolve the installed kernel identity against ``expected_pin``.

    Fail-closed contract: unknown beats guessed. A compatible kernel

    * has the ``uniharness`` distribution installed,
    * loads from a source tree whose ``git rev-parse HEAD`` equals
      ``expected_pin`` (the full delta-inclusive fork commit), and
    * descends from :data:`VENDORED_UPSTREAM_REVISION`.

    Any other state is incompatible; callers must treat it as fatal when
    Agent v2 is enabled (fail closed), never as a warning.
    """
    # 1. Installed distribution identity.
    try:
        version = importlib.metadata.version("uniharness")
    except importlib.metadata.PackageNotFoundError:
        return KernelIdentity(
            distribution_version=None,
            package_path=None,
            fork_commit=None,
            expected_pin=expected_pin,
            compatible=False,
            detail=(
                "UniHarness distribution is not installed; install the "
                f"vendored fork pinned at {VENDORED_UPSTREAM_REVISION[:12]} "
                f"(see {FORK_PROVENANCE_DOC})"
            ),
        )

    # 2. Source path of the loaded package (not site-packages metadata).
    try:
        import uniharness  # noqa: PLC0415 - deferred so disabled deploys skip it

        pkg_root = Path(uniharness.__file__).resolve().parent
    except (ImportError, AttributeError):
        return KernelIdentity(
            distribution_version=version,
            package_path=None,
            fork_commit=None,
            expected_pin=expected_pin,
            compatible=False,
            detail="UniHarness metadata exists but the package cannot be imported",
        )

    # 3. Fork commit of the source tree backing the loaded package.
    fork_commit = _resolve_fork_commit(pkg_root)

    if fork_commit is None:
        return KernelIdentity(
            distribution_version=version,
            package_path=str(pkg_root),
            fork_commit=None,
            expected_pin=expected_pin,
            compatible=False,
            detail=(
                "Installed uniharness package has no resolvable git fork "
                "commit; it does not provably come from the pinned fork"
            ),
        )

    if not fork_commit.startswith(expected_pin) and not expected_pin.startswith(fork_commit):
        return KernelIdentity(
            distribution_version=version,
            package_path=str(pkg_root),
            fork_commit=fork_commit,
            expected_pin=expected_pin,
            compatible=False,
            detail=f"Loaded fork commit {fork_commit} does not match configured pin",
        )

    upstream_is_ancestor = _pin_is_ancestor_of_head(pkg_root, VENDORED_UPSTREAM_REVISION)
    if upstream_is_ancestor is not True:
        return KernelIdentity(
            distribution_version=version,
            package_path=str(pkg_root),
            fork_commit=fork_commit,
            expected_pin=expected_pin,
            compatible=False,
            detail=(
                "Loaded source tree does not descend from the vendored "
                f"upstream revision {VENDORED_UPSTREAM_REVISION[:12]}"
            ),
        )

    return KernelIdentity(
        distribution_version=version,
        package_path=str(pkg_root),
        fork_commit=fork_commit,
        expected_pin=expected_pin,
        compatible=True,
        detail=f"Pinned fork verified at commit {fork_commit} (upstream ancestor ok)",
    )
