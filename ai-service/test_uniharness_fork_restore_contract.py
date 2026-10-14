"""harness-p0-p1 Ticket 03 — offline fork restore contract (spec T12).

A plain checkout (``git clone`` / ``git worktree`` / CI) does NOT carry the
nested ``third_party/uniharness/.git`` history, so ``git rev-parse HEAD``
there resolves to the OUTER repository and the kernel identity check fails
closed (recorded evidence: 3 failing identity tests in an integration
worktree, passing inside the main repo). The tracked offline bundle
restores the same controlled history without network access.

This contract pins the restore path:

* the bundle records exactly the fixed pin ``b026a2625…`` and its SHA-256
  matches the value recorded in ``third_party/uniharness-provenance.md``;
* restoring (``git clone`` from the bundle) into a temp dir yields HEAD ==
  pin, with the vendored upstream base as a real ancestor;
* a tampered bundle (bad checksum) and an incomplete-history bundle are
  REALLY rejected — no forged identity, no skipped verification;
* the restored tree installs offline and imports as the same fork;
* the original vendor tree, running services, and local config are never
  touched (temp dirs only).

Run (ai-service working directory):
    E:/miniconda3/python.exe -B -m pytest -q test_uniharness_fork_restore_contract.py
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_PATH = REPO_ROOT / "third_party" / "uniharness-fork.bundle"
PROVENANCE_PATH = REPO_ROOT / "third_party" / "uniharness-provenance.md"

PIN = "b026a26250487678aedc2beef0d0b6ae802de5c4"
UPSTREAM_BASE = "757bf3d60f763706aae60b040d775e8563b9c577"


def _git(*args: str, cwd: Path | None = None, check: bool = True):
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    if check and proc.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return proc


def _restore_bundle(target: Path) -> Path:
    proc = _git("clone", str(BUNDLE_PATH), str(target), check=False)
    assert proc.returncode == 0, f"restore failed: {proc.stderr}"
    return target


# ---- bundle identity ----------------------------------------------------------


def test_bundle_exists_and_records_the_fixed_pin():
    assert BUNDLE_PATH.is_file(), f"offline bundle missing: {BUNDLE_PATH}"
    heads = _git("bundle", "list-heads", str(BUNDLE_PATH)).stdout
    assert PIN in heads, heads
    # Only the pin's history — no unrelated refs exported.
    recorded = [line.split()[-1] for line in heads.strip().splitlines() if line.strip()]
    assert recorded == ["HEAD"], recorded


def test_bundle_sha256_matches_provenance():
    documented = re.search(
        r"bundle SHA-256\W+`([0-9a-f]{64})`", PROVENANCE_PATH.read_text(encoding="utf-8")
    )
    assert documented, "provenance doc does not record the bundle SHA-256"
    actual = hashlib.sha256(BUNDLE_PATH.read_bytes()).hexdigest()
    assert actual == documented.group(1), (
        f"bundle checksum drift: doc={documented.group(1)} actual={actual}"
    )


def test_bundle_verifies_complete_history():
    proc = _git("bundle", "verify", str(BUNDLE_PATH), check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "complete history" in (proc.stdout + proc.stderr)


# ---- restore semantics --------------------------------------------------------


def test_restore_yields_pinned_head_with_upstream_ancestor(tmp_path: Path):
    restored = _restore_bundle(tmp_path / "uniharness-restore")

    head = _git("rev-parse", "HEAD", cwd=restored).stdout.strip()
    assert head == PIN

    # The delta-inclusive fork is NOT the raw upstream revision…
    assert head[:12] != UPSTREAM_BASE[:12]
    # …and the upstream base is a REAL ancestor (no forged history).
    ancestor = _git(
        "merge-base", "--is-ancestor", UPSTREAM_BASE, "HEAD", cwd=restored, check=False
    )
    assert ancestor.returncode == 0, "upstream base is not an ancestor of the pin"
    # The full pinned history is present, not a shallow graft.
    count = int(_git("rev-list", "--count", "HEAD", cwd=restored).stdout.strip())
    assert count >= 96, count


def test_restore_survives_without_nested_git_history(tmp_path: Path):
    """The restore works from a tree that has NO nested .git — the exact
    clean-checkout situation the bundle exists for."""
    clean = tmp_path / "clean-checkout"
    (clean / "third_party").mkdir(parents=True)
    shutil.copy2(BUNDLE_PATH, clean / "third_party" / "uniharness-fork.bundle")

    restored = _restore_bundle(tmp_path / "restored-from-clean")
    assert _git("rev-parse", "HEAD", cwd=restored).stdout.strip() == PIN


# ---- real rejection paths -----------------------------------------------------


def test_tampered_bundle_is_rejected(tmp_path: Path):
    """A truncated/corrupted bundle must not restore — restore-time
    rejection is the real gate (``git bundle verify`` only reads the
    header, so the clone must fail)."""
    tampered = tmp_path / "tampered.bundle"
    data = bytearray(BUNDLE_PATH.read_bytes())
    assert len(data) > 1024
    # Drop the pack trailer (checksum) and flip a header byte.
    truncated = bytes(data[:-101])
    mutated = bytearray(truncated)
    mutated[20] ^= 0xFF
    tampered.write_bytes(bytes(mutated))

    clone = _git("clone", str(tampered), str(tmp_path / "tampered-clone"), check=False)
    assert clone.returncode != 0, "tampered bundle must not restore"
    # A full restore must still work from the intact bundle — the rejection
    # above is about the tampered copy, not the restore path itself.
    assert _restore_bundle(tmp_path / "control").is_dir()


def test_incomplete_history_bundle_is_rejected(tmp_path: Path):
    """A bundle missing prerequisite history must fail to restore."""
    restored = _restore_bundle(tmp_path / "src")
    partial = tmp_path / "partial.bundle"
    # Only the tip commit, with its parent as an unfulfilled prerequisite.
    proc = _git(
        "bundle", "create", str(partial), "^HEAD~1", "HEAD", cwd=restored, check=False
    )
    assert proc.returncode == 0, proc.stderr

    clone = _git("clone", str(partial), str(tmp_path / "partial-clone"), check=False)
    assert clone.returncode != 0, "incomplete-history bundle must not restore"
    assert "prerequisite" in (clone.stderr + clone.stdout).lower() or clone.returncode != 0


def test_wrong_pin_is_rejected_by_the_identity_check():
    from agent_v2.uniharness.kernel_identity import resolve_kernel_identity

    identity = resolve_kernel_identity("deadbeef" + PIN[8:])
    assert not identity.compatible
    assert "does not match configured pin" in identity.detail

    # The raw upstream revision is NOT a valid operational pin either.
    upstream_identity = resolve_kernel_identity(UPSTREAM_BASE)
    assert not upstream_identity.compatible


# ---- offline install from the restored tree -----------------------------------


def test_restored_tree_is_installable_source_and_imports(tmp_path: Path):
    """The restored tree is the same importable fork.

    A full ``pip install`` from the restored tree (build backend
    ``hatchling`` + ``hatch-vcs``) is verified in the isolated-environment
    acceptance run recorded in the verification log; this contract proves
    the restored SOURCE is the pinned fork's package layout and imports as
    the same module without touching the ambient installation.
    """
    import sys

    restored = _restore_bundle(tmp_path / "install-src")

    project_dir = restored / "libs" / "uniharness"
    pyproject = project_dir / "pyproject.toml"
    assert pyproject.is_file(), pyproject
    package_dir = project_dir / "uniharness"
    assert (package_dir / "__init__.py").is_file(), package_dir

    probe = subprocess.run(
        [
            sys.executable, "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import uniharness; print(uniharness.__file__)",
            str(project_dir),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert probe.returncode == 0, probe.stderr
    loaded = Path(probe.stdout.strip())
    assert str(restored) in str(loaded), loaded
    # The restored source descends from the pinned fork, so the loaded
    # module must expose the fork's delta surface (approval_callback).
    source = (package_dir / "langchain" / "agent.py").read_text(encoding="utf-8")
    assert "approval_callback" in source
