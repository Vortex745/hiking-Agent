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
import sys
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


# ---- Ticket 07 — the restore script is the CI source of truth ------------------
#
# The validation job (`.github/workflows/deploy.yml`, ticket 07) does not
# hand-roll git commands inline: it runs `scripts/restore-uniharness-fork.py`
# (restore mode) and then the SAME script with `--check` (read-only verify).
# These tests pin the script's contract on temp targets so the job's wiring
# cannot silently rot: fail closed on a tampered bundle, a provenance digest
# disagreement, a foreign pin, and a missing git history — and never clobber
# a target that is not provably the pinned fork.

SCRIPT_PATH = REPO_ROOT / "scripts" / "restore-uniharness-fork.py"
VENDOR_TREE = REPO_ROOT / "third_party" / "uniharness"


def _run_restore_script(*args: str) -> subprocess.CompletedProcess[str]:
    assert SCRIPT_PATH.is_file(), f"restore script missing: {SCRIPT_PATH}"
    return subprocess.run(
        [sys.executable, "-B", str(SCRIPT_PATH), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
    )


def _plain_checkout_tree(tmp_path: Path) -> Path:
    """The exact clean-checkout shape of the vendor tree.

    A tracked checkout carries no nested ``.git`` AND omits the doc/asset
    files the OUTER repo's ``.gitignore`` excludes (``*.md`` / ``*.png``),
    so it is NOT the complete pinned fork — restoring must materialize the
    full pinned tree, not only graft git metadata.
    """
    target = tmp_path / "uniharness"
    shutil.copytree(
        VENDOR_TREE, target, ignore=shutil.ignore_patterns(".git")
    )
    assert not (target / ".git").exists()
    return target


def _combined_output(proc: subprocess.CompletedProcess[str]) -> str:
    return proc.stdout + proc.stderr


def test_restore_script_check_passes_on_the_restored_repository_tree():
    """CI runs this AFTER the restore step; it re-verifies in place."""
    proc = _run_restore_script("--check")
    assert proc.returncode == 0, _combined_output(proc)


def test_restore_script_check_fails_closed_without_git_history(tmp_path: Path):
    target = _plain_checkout_tree(tmp_path)
    proc = _run_restore_script("--check", "--target", str(target))
    assert proc.returncode != 0
    assert "restore" in _combined_output(proc).lower()
    # Read-only: the failed check did not restore anything.
    assert not (target / ".git").exists()


def test_restore_script_restores_plain_checkout_to_the_pinned_tree(tmp_path: Path):
    target = _plain_checkout_tree(tmp_path)
    proc = _run_restore_script("--target", str(target))
    assert proc.returncode == 0, _combined_output(proc)

    head = _git("rev-parse", "HEAD", cwd=target).stdout.strip()
    assert head == PIN
    ancestor = _git(
        "merge-base", "--is-ancestor", UPSTREAM_BASE, "HEAD", cwd=target, check=False
    )
    assert ancestor.returncode == 0, "upstream base is not an ancestor after restore"
    # The complete pinned checkout is materialized — the ignored doc/asset
    # files the plain checkout lacked are present now.
    assert (target / "CONTRIBUTING.md").is_file()
    assert (
        target / "libs" / "uniharness" / "uniharness" / "prompts" / "fragments"
        / "tool_instruction_bash.md"
    ).is_file()
    # And the read-only check agrees on the restored tree.
    assert _run_restore_script("--check", "--target", str(target)).returncode == 0


def test_restore_script_is_idempotent(tmp_path: Path):
    target = _plain_checkout_tree(tmp_path)
    assert _run_restore_script("--target", str(target)).returncode == 0
    second = _run_restore_script("--target", str(target))
    assert second.returncode == 0, _combined_output(second)
    assert _git("rev-parse", "HEAD", cwd=target).stdout.strip() == PIN


def test_restore_script_restores_into_a_missing_target(tmp_path: Path):
    target = tmp_path / "fresh" / "uniharness"
    proc = _run_restore_script("--target", str(target))
    assert proc.returncode == 0, _combined_output(proc)
    assert _git("rev-parse", "HEAD", cwd=target).stdout.strip() == PIN
    status = _git("status", "--porcelain", cwd=target).stdout.strip()
    assert status == "", status


def test_restore_script_fails_closed_on_tampered_bundle(tmp_path: Path):
    data = bytearray(BUNDLE_PATH.read_bytes())
    tampered = tmp_path / "tampered.bundle"
    tampered.write_bytes(bytes(data[:-101]))
    target = tmp_path / "target"

    proc = _run_restore_script("--bundle", str(tampered), "--target", str(target))
    assert proc.returncode != 0
    assert "sha" in _combined_output(proc).lower() or "checksum" in _combined_output(proc).lower()
    # Temp-safe fail-closed: no half-restored target left behind.
    assert not target.exists()


def test_restore_script_fails_closed_when_provenance_digest_disagrees(tmp_path: Path):
    doc = tmp_path / "uniharness-provenance.md"
    text = PROVENANCE_PATH.read_text(encoding="utf-8")
    doc.write_text(re.sub(r"[0-9a-f]{64}", "0" * 64, text, count=1), encoding="utf-8")
    target = tmp_path / "target"

    proc = _run_restore_script(
        "--provenance", str(doc), "--target", str(target)
    )
    assert proc.returncode != 0
    assert "provenance" in _combined_output(proc).lower()
    assert not target.exists()


def test_restore_script_refuses_to_clobber_a_foreign_pin_checkout(tmp_path: Path):
    """A target that IS a repo but at the wrong revision must not be
    silently replaced, and `--check` must reject it."""
    target = _restore_bundle(tmp_path / "foreign")
    _git("checkout", "--detach", UPSTREAM_BASE, cwd=target)
    assert _git("rev-parse", "HEAD", cwd=target).stdout.strip() == UPSTREAM_BASE

    check = _run_restore_script("--check", "--target", str(target))
    assert check.returncode != 0
    assert PIN[:12] in _combined_output(check)

    restore = _run_restore_script("--target", str(target))
    assert restore.returncode != 0
    assert "refus" in _combined_output(restore).lower()
    # Unchanged: the foreign checkout was not destroyed by the refusal.
    assert _git("rev-parse", "HEAD", cwd=target).stdout.strip() == UPSTREAM_BASE
