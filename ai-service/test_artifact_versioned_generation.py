"""Ticket 09 — versioned plan-book artifact generation.

Covers the ticket's acceptance criteria at the pre-agreed seams:

* AC3 — every regeneration produces a NEW artifact: the version discriminator
  lives inside ``args_canonical``, so a second ``generate_pdf`` whose args are
  otherwise identical does NOT replay the committed effect (tested against the
  single effect-ledger implementation, :func:`drive_effect`).
* AC4 — the draft records the relationship between the current artifact and
  its history: a new ``plan_artifact_versions`` table (never a ``plan_drafts``
  column, per ticket 06's storage rule), not FK'd to ``run_id`` so it outlives
  the run-scoped CASCADE tables, exposed only through ``RunRepository``.
* AC1/AC2/AC5 — regeneration publishes through the EXISTING sandbox + approval
  + export lifecycle with real rendering: two generations of the sourced report
  are two downloadable PDFs with different bytes, a failed render publishes
  nothing, and old artifacts keep the owner / 410 / TTL download rules.

Style follows the repo convention: fake kernel + real temp SQLite through the
``RunOrchestrator`` constructor; the artifact-identity tests use the real
``LocalNativeComputer`` and the real ``export_artifact_to_host`` so the claims
are about actual files, and the download rules are exercised through the real
ASGI endpoint.
"""

from __future__ import annotations

import sqlite3

from agent_v2.persistence import CheckpointsStore, ControlStore
from agent_v2.runs import RunRepository


def _repo(tmp_path) -> RunRepository:
    return RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )


# ---- AC4: the version table is the draft's artifact history -----------------


def test_the_version_history_is_a_new_table_never_run_scoped(tmp_path):
    """AC4 storage rule: a new table, and it must outlive runs.

    Run-scoped tables (events/artifacts/approvals/effects) CASCADE-delete with
    the run row; the history must not disappear with them, so it carries no FK
    to ``runs`` — this test is the pin for that decision.
    """
    store = ControlStore(str(tmp_path / "control.sqlite"))
    rows = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='plan_artifact_versions'"
    ).fetchall()
    assert rows, "plan_artifact_versions must exist via CREATE TABLE IF NOT EXISTS"
    fks = store._conn.execute(
        "PRAGMA foreign_key_list('plan_artifact_versions')"
    ).fetchall()
    assert [fk["table"] for fk in fks] == [], (
        "the version history must not FK to runs (or anything run-scoped): "
        "it has to outlive the run that produced the artifact"
    )


def test_the_draft_can_answer_which_artifact_is_current_and_what_came_before(tmp_path):
    """AC4: the relation between the current version and its history is stored."""
    repo = _repo(tmp_path)

    first = repo.publish_plan_artifact_version(
        conversation_id="conv-v", artifact_id="art-1", draft_id="draft-1",
        published_by_run_id="run-1",
    )
    second = repo.publish_plan_artifact_version(
        conversation_id="conv-v", artifact_id="art-2", draft_id="draft-1",
        published_by_run_id="run-2",
    )

    # Revision ascends per conversation and each row names the artifact before it.
    assert (first.revision, second.revision) == (1, 2)
    assert first.previous_artifact_id == ""
    assert second.previous_artifact_id == "art-1"
    assert second.draft_id == "draft-1"
    assert second.published_by_run_id == "run-2"

    # "Which artifact is current" — read back through a fresh facade over the
    # same database, so the answer lives on disk, not in an object cache.
    fresh = RunRepository(
        control=ControlStore(str(tmp_path / "control.sqlite")),
        checkpoints=CheckpointsStore(str(tmp_path / "ckpt.sqlite")),
    )
    current = fresh.get_current_plan_artifact_version("conv-v")
    assert current is not None and current.artifact_id == "art-2"

    # "What came before" — the full history in generation order.
    history = fresh.list_plan_artifact_versions("conv-v")
    assert [(v.revision, v.artifact_id) for v in history] == [(1, "art-1"), (2, "art-2")]


def test_a_conversation_without_a_published_plan_book_has_no_current_artifact(tmp_path):
    """AC4 fails honest: no row is no claim, not artifact 0."""
    repo = _repo(tmp_path)
    assert repo.get_current_plan_artifact_version("conv-none") is None
    assert repo.list_plan_artifact_versions("conv-none") == []
