#!/usr/bin/env python3
"""Restore / validate the vendored UniHarness fork from the offline bundle.

harness-p0-p1 ticket 07 (spec T13). The validation job
(``.github/workflows/deploy.yml``) uses this script as its single source of
truth for fork restore + identity validation; no CI step hand-rolls git
commands.

A plain checkout (``git clone`` / ``git worktree`` / CI) carries neither the
nested ``third_party/uniharness/.git`` history nor the doc/asset files the
OUTER repository's ``.gitignore`` excludes (``*.md`` / ``*.png``), so
``git rev-parse HEAD`` resolves to the OUTER repository and the kernel
identity check fails closed. The tracked bundle
``third_party/uniharness-fork.bundle`` restores the same controlled history
WITHOUT any network access.

Semantics (verified independently before this script existed):

* ``git bundle verify`` must pass and the bundle's SHA-256 must equal the
  value recorded in ``third_party/uniharness-provenance.md``;
* clone via ``GIT_ALLOW_PROTOCOL=file`` / ``GIT_TERMINAL_PROMPT=0`` (no
  network protocol can be used, no credential prompt can appear);
* restored HEAD must equal the fixed pin, the vendored upstream revision
  must be a real ancestor of it, and the full pinned history must be
  present;
* the restored tree is materialized in a temp-safe way: the bundle is
  cloned and validated in a staging directory first, and the target is only
  touched after that validation passes;
* fail closed with clear messages — a tampered bundle, a provenance digest
  disagreement, a missing/foreign target checkout, or leftover junk must
  never be silently "fixed" into a passing state.

Usage::

    python scripts/restore-uniharness-fork.py            # restore (idempotent)
    python scripts/restore-uniharness-fork.py --check    # read-only verify

Exit code 0 = verified; 1 = fail closed (details on stderr).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_BUNDLE = REPO_ROOT / "third_party" / "uniharness-fork.bundle"
DEFAULT_TARGET = REPO_ROOT / "third_party" / "uniharness"
DEFAULT_PROVENANCE = REPO_ROOT / "third_party" / "uniharness-provenance.md"

# Fixed operational pin (delta HEAD) and its vendored upstream base.
PIN = "aeceee7ea5bdb8c4dcaa7d1c54400afacabe5556"
UPSTREAM_BASE = "757bf3d60f763706aae60b040d775e8563b9c577"
# The pin's reachable history (upstream base + five controlled delta commits
# + their ancestry); a shallow/grafted history is rejected.
MIN_COMMITS = 97

_SHA256_IN_DOC = re.compile(r"bundle SHA-256\W+`([0-9a-f]{64})`")


class RestoreError(RuntimeError):
    """Fail-closed restore/check failure with an operator-readable message."""


def _git(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    # No network protocol may be used and no credential prompt may appear:
    # the bundle is a local file and everything must stay offline.
    env["GIT_ALLOW_PROTOCOL"] = "file"
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        return subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", *args],
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise RestoreError("git executable not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise RestoreError(f"git {' '.join(args)} timed out after {timeout}s") from exc


def _git_ok(args: list[str], *, cwd: Path | None = None, what: str) -> str:
    proc = _git(args, cwd=cwd)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise RestoreError(f"{what} failed (git {' '.join(args)} rc={proc.returncode}): {detail}")
    return proc.stdout


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recorded_bundle_sha256(provenance: Path) -> str:
    if not provenance.is_file():
        raise RestoreError(f"provenance file is missing: {provenance}")
    match = _SHA256_IN_DOC.search(provenance.read_text(encoding="utf-8"))
    if match is None:
        raise RestoreError(
            f"provenance file does not record the bundle SHA-256: {provenance}"
        )
    return match.group(1)


def verify_bundle(bundle: Path, provenance: Path) -> str:
    """Verify bundle integrity and that it records exactly the fixed pin."""
    if not bundle.is_file():
        raise RestoreError(f"offline bundle is missing: {bundle}")

    recorded = _recorded_bundle_sha256(provenance)
    actual = _sha256_file(bundle)
    if actual != recorded:
        raise RestoreError(
            "bundle SHA-256 does not match the provenance record "
            f"(provenance={recorded} actual={actual}) — restore refused"
        )
    print(f"[bundle] sha256 {actual} matches provenance")

    proc = _git(["bundle", "verify", str(bundle)])
    if proc.returncode != 0:
        detail = (proc.stdout + proc.stderr).strip()
        raise RestoreError(f"git bundle verify failed: {detail}")
    print("[bundle] git bundle verify OK")

    heads_out = _git_ok(
        ["bundle", "list-heads", str(bundle)], what="git bundle list-heads"
    )
    heads = {}
    for line in heads_out.splitlines():
        parts = line.split()
        if len(parts) == 2:
            heads[parts[1]] = parts[0]
    if PIN not in heads.values():
        raise RestoreError(
            f"bundle does not record the fixed pin {PIN}: heads={heads}"
        )
    if "HEAD" in heads and heads["HEAD"] != PIN:
        raise RestoreError(
            f"bundle HEAD is {heads['HEAD']}, expected the fixed pin {PIN}"
        )
    print(f"[bundle] records the fixed pin {PIN}")
    return actual


def verify_target(target: Path) -> None:
    """Verify that ``target`` is a complete checkout of the pinned fork."""
    if not target.is_dir():
        raise RestoreError(
            f"UniHarness fork is not restored at {target}; "
            "run this script without --check to restore the pinned fork"
        )
    # The restored fork MUST carry its own nested ``.git``: inside a plain
    # checkout (repo worktree / CI) ``git rev-parse`` would walk up to the
    # OUTER repository and report the product repo's HEAD, which is exactly
    # the failure the offline bundle restore exists to fix.
    if not (target / ".git").exists():
        raise RestoreError(
            f"{target} is a plain checkout without its restored git history; "
            "run this script without --check to restore the pinned fork"
        )
    probe = _git(["rev-parse", "--git-dir"], cwd=target)
    if probe.returncode != 0:
        raise RestoreError(
            f"{target} has broken git metadata ({probe.stderr.strip()}); "
            "run this script without --check to restore the pinned fork"
        )

    head = _git_ok(["rev-parse", "HEAD"], cwd=target, what="git rev-parse HEAD").strip()
    if head != PIN:
        raise RestoreError(
            f"restored HEAD {head} does not equal the fixed pin {PIN} — "
            "refusing to accept a foreign UniHarness checkout"
        )

    ancestor = _git(
        ["merge-base", "--is-ancestor", UPSTREAM_BASE, "HEAD"], cwd=target
    )
    if ancestor.returncode != 0:
        raise RestoreError(
            "vendored upstream revision "
            f"{UPSTREAM_BASE} is not an ancestor of HEAD={head} — "
            "the restored history is incomplete or forged"
        )

    count = int(
        _git_ok(
            ["rev-list", "--count", "HEAD"], cwd=target, what="git rev-list --count"
        ).strip()
    )
    if count < MIN_COMMITS:
        raise RestoreError(
            f"restored history has {count} commits, fewer than the pinned "
            f"{MIN_COMMITS} — incomplete history refused"
        )

    status = _git_ok(
        ["status", "--porcelain"], cwd=target, what="git status --porcelain"
    )
    if status.strip():
        first = "\n".join(status.strip().splitlines()[:5])
        raise RestoreError(
            "restored tree is not the clean pinned checkout; leftover or "
            f"modified entries:\n{first}"
        )
    print(
        f"[target] {target} HEAD == pin, upstream ancestor OK, "
        f"{count} commits, working tree clean"
    )


def _has_own_git(target: Path) -> bool:
    """True when the target carries its OWN git metadata (not an outer repo).

    ``git rev-parse --git-dir`` is not enough: inside a plain checkout it
    walks up to the OUTER repository. Only a nested ``.git`` (directory or
    worktree gitfile) proves the fork's own history.
    """
    return (target / ".git").exists()


def restore(bundle: Path, provenance: Path, target: Path) -> None:
    """Restore the pinned fork into ``target`` (idempotent, temp-safe)."""
    verify_bundle(bundle, provenance)

    if target.exists() and _has_own_git(target):
        # Already a repo: either it already IS the pin (idempotent no-op) or
        # it is something else — never clobber the latter.
        try:
            verify_target(target)
        except RestoreError as exc:
            raise RestoreError(
                f"{target} is a git checkout but not the pinned fork; "
                f"refusing to clobber it ({exc})"
            ) from exc
        print("[restore] already at the pinned fork — nothing to do")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=".uniharness-restore-", dir=str(target.parent))
    )
    staging = staging_root / "uniharness"
    try:
        # Clone + validate in staging first: the target is only touched after
        # the bundle is proven to be the pinned fork (temp-safe).
        _git_ok(
            ["clone", str(bundle), str(staging)],
            what="git clone from the offline bundle",
        )
        verify_target(staging)
        print(f"[restore] staging clone validated at {staging}")

        existed = target.exists()
        if existed:
            # Hydrate the existing plain checkout in place: copy the complete
            # pinned tree over it (the checkout may lack .gitignore'd files),
            # then graft the validated git metadata. Replace only file/symlink
            # type collisions at paths from the staged tree; retain other files.
            for source_path in staging.rglob("*"):
                relative_path = source_path.relative_to(staging)
                if relative_path.parts[0] == ".git":
                    continue
                target_path = target / relative_path
                if source_path.is_symlink():
                    if target_path.is_symlink() or target_path.is_file():
                        target_path.unlink()
                    elif target_path.exists():
                        raise RestoreError(
                            f"cannot restore symlink {relative_path}: "
                            "target path is a directory"
                        )
                elif target_path.is_symlink():
                    target_path.unlink()
            shutil.copytree(
                staging,
                target,
                dirs_exist_ok=True,
                symlinks=True,
                ignore=shutil.ignore_patterns(".git"),
            )
            shutil.move(str(staging / ".git"), str(target / ".git"))
            try:
                verify_target(target)
            except RestoreError:
                # Roll back the grafted metadata, keep the hydrated files.
                shutil.rmtree(target / ".git", ignore_errors=True)
                raise
        else:
            shutil.move(str(staging), str(target))
            verify_target(target)
        print(f"[restore] UniHarness fork restored at {target} (pin {PIN})")
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restore (default) or verify (--check) the vendored UniHarness "
            "fork from third_party/uniharness-fork.bundle — offline, "
            "fail-closed."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="read-only verification: nothing is restored or modified",
    )
    parser.add_argument(
        "--bundle", type=Path, default=DEFAULT_BUNDLE, help="override bundle path"
    )
    parser.add_argument(
        "--target", type=Path, default=DEFAULT_TARGET, help="override restore target"
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=DEFAULT_PROVENANCE,
        help="override provenance document path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    bundle = args.bundle.resolve()
    target = args.target.resolve()
    provenance = args.provenance.resolve()
    try:
        if args.check:
            verify_bundle(bundle, provenance)
            verify_target(target)
            print("[check] UniHarness fork restore verified (read-only)")
        else:
            restore(bundle, provenance, target)
    except RestoreError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
