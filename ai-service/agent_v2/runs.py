"""Ticket 06 — Run lifecycle facade.

Composes :class:`ControlStore` + :class:`CheckpointsStore` into a
single API the endpoint layer consumes. The facade hides the two-database
split so ``api/chat_v2.py`` deals with one object: ``RunRepository``.

Responsibilities:

* Idempotent run creation — same ``request_id`` returns the same run
  without re-emitting user messages or re-invoking tools (spec AC2).
* Run status transitions — ``running`` → ``done``/``error``/``cancelled``.
  Terminal runs reject further transitions (spec AC5).
* Snapshot assembly — conversation + status + sequence cursor + pending
  actions + user-recoverable content, no raw framework events (spec AC3).
* Event replay — return events with sequence > ``from_seq`` for
  reconnect (spec AC4).
* Idempotent cancel — repeat cancels return the original outcome
  (spec AC5).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from agent_v2.persistence import (
    ApprovalRecord,
    ArtifactRecord,
    CheckpointsStore,
    ControlStore,
    EffectRecord,
    RunAlreadyTerminal,
    RunNotFound,
    RunRecord,
    SandboxLeaseRecord,
)

logger = logging.getLogger("ai-service.agent_v2.runs")

# Ticket 05 (T07): the bounded window for the model-visible conversation
# history. The limits apply to the HISTORY only — the current user message
# is appended by the kernel afterwards and never counts against them.
HISTORY_MAX_MESSAGES = 20
HISTORY_MAX_CHARS = 8000

# Newest-first page size for the history scan. The scan stops as soon as
# the next older completed turn cannot fit the window, so a window-sized
# conversation is read with one or two pages.
_HISTORY_PAGE_SIZE = 50


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    """Spec AC3 snapshot payload.

    Fields are deliberately product-facing — no raw framework events,
    no internal store paths. ``pending_actions`` exposes tool_calls
    that started but did not complete before the stream closed.
    """

    run_id: str
    request_id: str
    conversation_id: str
    profile_id: str
    status: str
    terminal_type: str | None
    created_at: str
    updated_at: str
    terminal_at: str | None
    sequence_cursor: int
    messages: list[dict[str, Any]]
    pending_actions: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class CancelOutcome:
    """Spec AC5 cancel result."""

    accepted: bool
    reason: str
    run_id: str


class RunRepository:
    """Facade over ControlStore + CheckpointsStore.

    The two stores live in separate SQLite files (spec decision #40)
    but are composed here so the endpoint layer doesn't need to know
    the split. Operations that span both stores (e.g. create_run
    inserts into control.runs + checkpoints.conversations) are wrapped
    in a single method — partial failure leaves the system recoverable
    because the run_id is the join key and missing conversation rows
    are recreated on the next message.
    """

    def __init__(self, control: ControlStore, checkpoints: CheckpointsStore) -> None:
        self.control = control
        self.checkpoints = checkpoints

    def close(self) -> None:
        self.control.close()
        self.checkpoints.close()

    # -- create + status ------------------------------------------------

    def create_run_idempotent(
        self,
        *,
        request_id: str,
        conversation_id: str,
        profile_id: str,
        user_message: str,
    ) -> tuple[str, bool]:
        """Create a run + persist the user message, idempotent on request_id.

        Returns ``(run_id, created)``. When ``created=False`` the run
        already exists from a prior call with the same request_id — the
        user message is NOT re-appended and no model/tool call happens
        (spec AC2).

        The conversation row is upserted so a returning user (same
        conversation_id) extends the same history across runs.
        """
        run_id, created = self.control.create_run_idempotent(
            request_id=request_id,
            conversation_id=conversation_id,
            profile_id=profile_id,
        )
        if created:
            # Only persist user message for a fresh run — replays skip.
            self.checkpoints.ensure_conversation(
                conversation_id=conversation_id,
                profile_id=profile_id,
            )
            # Sequence 0 = user input emitted before the first v2 event.
            # Real v2 events start at sequence 1 (per EventStream).
            self.checkpoints.append_message(
                conversation_id=conversation_id,
                run_id=run_id,
                role="user",
                content=user_message,
                sequence=0,
            )
        return run_id, created

    def get_run(self, run_id: str) -> RunRecord | None:
        return self.control.get_run(run_id)

    def list_paused_runs_updated_before(self, updated_before: str) -> list[RunRecord]:
        """List abandoned paused runs — paused-run janitor support.

        See :meth:`ControlStore.list_paused_runs_updated_before`.
        """
        return self.control.list_paused_runs_updated_before(updated_before)

    def resolve_run_id(self, identifier: str) -> str | None:
        """Resolve an identifier to a real run_id.

        Tries ``run_id`` first, then falls back to ``request_id``. This
        supports early cancel (spec AC4): when the user cancels before
        the first v2 event has delivered ``run_id`` to the client, the
        frontend only knows ``request_id`` and sends it as the path
        parameter. Without this fallback the server returns
        ``unknown_run`` and the run keeps running on the server.
        """
        run = self.control.get_run(identifier)
        if run is not None:
            return run.run_id
        run = self.control.get_run_by_request(identifier)
        if run is not None:
            return run.run_id
        return None

    # -- sandbox leases (Ticket 09) -----------------------------------

    def create_sandbox_lease(
        self,
        *,
        run_id: str,
        sandbox_type: str,
        workspace_path: str,
    ) -> str:
        """Persist a sandbox lease for ``run_id`` (spec AC1).

        Delegates to :class:`ControlStore`. The caller (stream endpoint)
        creates the workspace via :class:`Computer` first, then records
        the lease. A run may have at most one lease.
        """
        return self.control.create_sandbox_lease(
            run_id=run_id,
            sandbox_type=sandbox_type,
            workspace_path=workspace_path,
        )

    def get_sandbox_lease(self, run_id: str) -> SandboxLeaseRecord | None:
        """Return the sandbox lease bound to ``run_id``, if any."""
        return self.control.get_sandbox_lease(run_id)

    def destroy_sandbox_lease(
        self,
        *,
        run_id: str,
        resource_usage: dict[str, Any] | None = None,
        failure_reason: str | None = None,
    ) -> SandboxLeaseRecord | None:
        """Mark the lease destroyed (clean) or failed (cleanup error).

        Spec AC6: run completion or cancellation triggers sandbox cleanup.
        The lease records success/failure + resource usage so operators
        can audit cost and investigate stale sandboxes.
        """
        return self.control.destroy_sandbox_lease(
            run_id=run_id,
            resource_usage=resource_usage,
            failure_reason=failure_reason,
        )

    # -- artifacts (Ticket 10) -----------------------------------------

    def insert_artifact(
        self,
        *,
        artifact_id: str,
        run_id: str,
        conversation_id: str,
        public_filename: str,
        mime_type: str,
        size_bytes: int,
        sha256: str,
        storage_path: str,
        status: str,
        failure_reason: str | None,
        created_at: str,
        expires_at: str,
    ) -> ArtifactRecord:
        """Persist an exported artifact (Ticket 10).

        Spec AC3: ``artifact_id`` is opaque; ``storage_path`` never
        leaves the host. Spec AC4: a ``status='failed'`` row is allowed
        so evidence is retained before the sandbox is destroyed.
        """
        return self.control.insert_artifact(
            artifact_id=artifact_id,
            run_id=run_id,
            conversation_id=conversation_id,
            public_filename=public_filename,
            mime_type=mime_type,
            size_bytes=size_bytes,
            sha256=sha256,
            storage_path=storage_path,
            status=status,
            failure_reason=failure_reason,
            created_at=created_at,
            expires_at=expires_at,
        )

    def get_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        return self.control.get_artifact(artifact_id)

    def list_artifacts_for_run(self, run_id: str) -> list[ArtifactRecord]:
        return self.control.list_artifacts_for_run(run_id)

    def mark_artifact_expired(self, artifact_id: str) -> ArtifactRecord | None:
        return self.control.mark_artifact_expired(artifact_id)

    def mark_run_done(self, run_id: str) -> None:
        """Transition a running run to ``done`` terminal.

        Raises ``RunAlreadyTerminal`` if the run already terminated
        (e.g. cancel beat the stream to the transition). Callers that
        want idempotent semantics wrap this in ``_mark_terminal_safe``
        — the repository surfaces the constraint so direct callers
        know continuation was rejected (spec AC5).
        """
        self.control.update_run_status(
            run_id=run_id,
            status="done",
            terminal_type="done",
        )

    def mark_run_error(self, run_id: str) -> None:
        """Transition a running run to ``error`` terminal.

        Raises ``RunAlreadyTerminal`` if the run already terminated;
        see :meth:`mark_run_done` for the rationale.
        """
        self.control.update_run_status(
            run_id=run_id,
            status="error",
            terminal_type="error",
        )

    # -- paused/resume (Ticket 11) ------------------------------------

    def mark_run_paused(self, run_id: str) -> None:
        """Transition a running run to ``paused`` (spec AC3).

        ``paused`` is NOT terminal — the run is awaiting user approval
        and will resume via :meth:`resume_run`. The RunAlreadyTerminal
        guard allows this transition because ``paused`` is not in
        :data:`_TERMINAL_STATUSES`.

        Raises ``RunAlreadyTerminal`` if the run already reached a
        terminal state (done/error/cancelled) — a terminal run cannot
        pause because its lifecycle is over.
        """
        self.control.update_run_status(
            run_id=run_id,
            status="paused",
            terminal_type=None,
        )

    def resume_run(self, run_id: str) -> None:
        """Transition a paused run back to ``running`` (spec AC4).

        Called by the decision endpoint after the user approves or
        rejects the pending approval. The stream that follows reads the
        run's current state and continues from where it paused.

        Raises ``RunAlreadyTerminal`` if the run reached a terminal
        state while paused (e.g. cancel API was called, or an operator
        force-terminated the run). This is the spec AC5 invariant: a
        terminal run cannot be resumed.
        """
        self.control.update_run_status(
            run_id=run_id,
            status="running",
            terminal_type=None,
        )

    def is_paused(self, run_id: str) -> bool:
        """Check whether a run is paused awaiting approval."""
        run = self.control.get_run(run_id)
        if run is None:
            return False
        return run.status == "paused"

    # -- approvals (Ticket 11) -----------------------------------------

    def insert_approval(
        self,
        *,
        approval_id: str,
        run_id: str,
        conversation_id: str,
        tool_name: str,
        args_canonical: str,
        risk_level: str,
        impact: str,
        created_at: str,
        expires_at: str,
    ) -> ApprovalRecord:
        """Persist a new approval row in ``pending`` status (spec AC2).

        Delegates to :class:`ControlStore`. The approval id is computed
        deterministically in :mod:`agent_v2.approvals`; the DB UNIQUE
        constraint catches duplicate inserts at the layer of truth.
        """
        return self.control.insert_approval(
            approval_id=approval_id,
            run_id=run_id,
            conversation_id=conversation_id,
            tool_name=tool_name,
            args_canonical=args_canonical,
            risk_level=risk_level,
            impact=impact,
            created_at=created_at,
            expires_at=expires_at,
        )

    def get_approval(self, approval_id: str) -> ApprovalRecord | None:
        return self.control.get_approval(approval_id)

    def list_pending_approvals_for_run(self, run_id: str) -> list[ApprovalRecord]:
        return self.control.list_pending_approvals_for_run(run_id)

    def list_approvals_for_run(self, run_id: str) -> list[ApprovalRecord]:
        return self.control.list_approvals_for_run(run_id)

    def resolve_approval(
        self,
        *,
        approval_id: str,
        status: str,
        decision_request_id: str,
    ) -> ApprovalRecord | None:
        """Transition an approval to ``approved`` or ``rejected`` (spec AC4/AC5).

        Delegates to :class:`ControlStore`. The store enforces
        idempotency on ``decision_request_id`` and rejects transitions
        on already-terminal approvals — the first decision wins, so the
        tool executes at most once.
        """
        return self.control.resolve_approval(
            approval_id=approval_id,
            status=status,
            decision_request_id=decision_request_id,
        )

    def mark_approval_expired(self, approval_id: str) -> ApprovalRecord | None:
        return self.control.mark_approval_expired(approval_id)

    # -- effects (Ticket 12) -------------------------------------------

    def insert_effect(
        self,
        *,
        effect_key: str,
        run_id: str,
        conversation_id: str,
        tool_name: str,
        args_canonical: str,
        external_idempotency_key: str,
    ) -> EffectRecord:
        """Persist a new effect row in ``authorized`` status (spec AC1).

        Delegates to :class:`ControlStore`. The UNIQUE(run_id, tool_name,
        args_canonical) constraint catches duplicate inserts at the DB
        layer; callers that want idempotent semantics call
        :meth:`get_effect_by_tool_call` first (spec AC5).
        """
        return self.control.insert_effect(
            effect_key=effect_key,
            run_id=run_id,
            conversation_id=conversation_id,
            tool_name=tool_name,
            args_canonical=args_canonical,
            external_idempotency_key=external_idempotency_key,
        )

    def get_effect(self, effect_key: str) -> EffectRecord | None:
        return self.control.get_effect(effect_key)

    def get_effect_by_tool_call(
        self,
        *,
        run_id: str,
        tool_name: str,
        args_canonical: str,
    ) -> EffectRecord | None:
        return self.control.get_effect_by_tool_call(
            run_id=run_id,
            tool_name=tool_name,
            args_canonical=args_canonical,
        )

    def list_effects_for_run(self, run_id: str) -> list[EffectRecord]:
        return self.control.list_effects_for_run(run_id)

    def transition_effect_to_executing(
        self,
        effect_key: str,
    ) -> tuple[EffectRecord | None, bool]:
        """CAS transition ``authorized`` → ``executing`` (spec AC7).

        Returns ``(record, won)``. Only one concurrent caller sees
        ``won=True``; others see the already-``executing`` row with
        ``won=False`` and must NOT re-execute.
        """
        return self.control.transition_effect_to_executing(effect_key)

    def commit_effect(
        self,
        effect_key: str,
        result_json: str,
    ) -> EffectRecord | None:
        return self.control.commit_effect(effect_key, result_json)

    def mark_effect_outcome_unknown(
        self,
        effect_key: str,
        reason: str = "",
    ) -> EffectRecord | None:
        return self.control.mark_effect_outcome_unknown(effect_key, reason)

    def mark_effect_failed(
        self,
        effect_key: str,
        reason: str = "",
    ) -> EffectRecord | None:
        return self.control.mark_effect_failed(effect_key, reason)

    # -- events ---------------------------------------------------------

    def persist_event(
        self,
        *,
        run_id: str,
        sequence: int,
        event_type: str,
        payload: dict[str, Any],
        request_id: str,
        conversation_id: str,
        event_id: str | None = None,
        timestamp: str | None = None,
    ) -> None:
        """Append a v2 SSE event for replay.

        Spec AC4: sequence is monotonic per run. The UNIQUE(run_id,
        sequence) constraint enforces this at the DB layer — duplicate
        inserts raise IntegrityError, signaling a programming bug.

        ``event_id`` and ``timestamp`` are passed through from the wire
        envelope so the persisted row matches what the client saw (spec
        decision #26). When omitted, the store generates them.
        """
        self.checkpoints.append_event(
            run_id=run_id,
            sequence=sequence,
            event_type=event_type,
            payload=payload,
            request_id=request_id,
            conversation_id=conversation_id,
            event_id=event_id,
            timestamp=timestamp,
        )

    def get_events_since(
        self,
        *,
        run_id: str,
        from_seq: int,
    ) -> list[dict[str, Any]]:
        """Return events with sequence > from_seq as envelope dicts.

        Each dict matches the v2 wire format: ``version``, ``event_id``,
        ``sequence``, ``run_id``, ``conversation_id``, ``timestamp``,
        ``type``, ``request_id``, ``payload`` (spec decision #26).
        """
        stored = self.checkpoints.get_events_since(
            run_id=run_id,
            from_seq=from_seq,
        )
        return [
            {
                "version": "2",
                "event_id": s.event_id,
                "sequence": s.sequence,
                "run_id": s.run_id,
                "conversation_id": s.conversation_id,
                "timestamp": s.created_at,
                "type": s.event_type,
                "request_id": s.request_id,
                "payload": json.loads(s.payload_json),
            }
            for s in stored
        ]

    # -- bounded conversation history for the model (Ticket 05 / T07) ---

    def get_recent_completed_history(
        self,
        *,
        conversation_id: str,
        current_run_id: str | None = None,
        max_messages: int = HISTORY_MAX_MESSAGES,
        max_chars: int = HISTORY_MAX_CHARS,
    ) -> list[dict[str, str]]:
        """Recent completed conversation turns for the NEXT model turn.

        Server-side read: the caller passes the CURRENT run's
        conversation_id (and the run itself so it is excluded). Only
        ``user`` / ``assistant`` messages of naturally completed
        (``done``) runs qualify; running, failed, cancelled, paused and
        internal (tool/approval) turns never enter history. Messages are
        returned oldest-first in stable true insertion order (``rowid``);
        a turn is ONE whole run, ordered by the run's first inserted
        message, so runs whose messages interleave in storage (two
        concurrent completed runs of one conversation) still never split
        into pseudo-turns.

        The window keeps at most ``max_messages`` messages AND at most
        ``max_chars`` body characters, trimmed from the OLDEST end by
        COMPLETE turns: a turn is either fully present or fully absent —
        never half a fact. When a newer turn does not fit, older turns
        are omitted with it, so stale constraints can never outrank the
        newest ones. No completed history at all returns ``[]`` (fresh
        conversation semantics).

        Storage failures propagate to the caller, which maps them to the
        explicit ``context_unavailable`` terminal — a broken store must
        never be silently treated as a fresh conversation.
        """
        if not conversation_id:
            return []
        completed_run_ids = self.control.list_completed_run_ids(conversation_id)
        if not completed_run_ids:
            return []

        def _wanted(message: Any) -> bool:
            if message.role not in ("user", "assistant"):
                return False
            if message.run_id not in completed_run_ids:
                return False
            return not (current_run_id is not None and message.run_id == current_run_id)

        def _select(
            turns: list[list[Any]],
            *,
            last_resolved: bool,
        ) -> tuple[list[list[Any]], bool]:
            """Newest turns that fit; ``final`` = no older turn can change it.

            Stop at the FIRST turn that would overflow: an unresolved
            (page-boundary) turn can only grow, so it can never fit later
            either; and once a turn is dropped, every OLDER turn is
            dropped with it — stale constraints must never outrank the
            newest ones.
            """
            selected: list[list[Any]] = []
            used_messages = 0
            used_chars = 0
            for turn in turns:
                if not _wanted(turn[0]):
                    # Excluded by run status — skipped whole, independent
                    # of the window.
                    continue
                turn_chars = sum(len(message.content) for message in turn)
                if (
                    used_messages + len(turn) > max_messages
                    or used_chars + turn_chars > max_chars
                ):
                    return selected, True
                selected.append(turn)
                used_messages += len(turn)
                used_chars += turn_chars
            # Keep reading only while the oldest turn may still grow.
            return selected, not turns or last_resolved

        # Newest-first scan. EVERY message of one run belongs to ONE block
        # keyed by ``run_id`` — a run's messages are never split by other
        # runs interleaved between them, or trimming would keep half a
        # pair (an assistant answer without its user question). Blocks are
        # ordered (newest first) by the run's FIRST inserted message — the
        # run's start — so a later-starting turn stays last and is
        # trimmed first; a run whose OLDER message just arrived moves
        # back with it. The scan stops as soon as the next older block
        # cannot fit the window, so reading cost stays bounded with the
        # window.
        blocks: dict[str, list[Any]] = {}
        block_order: list[str] = []
        selected: list[list[Any]] = []
        offset = 0
        exhausted = False
        while True:
            page = self.checkpoints.get_messages_newest_first(
                conversation_id=conversation_id,
                limit=_HISTORY_PAGE_SIZE,
                offset=offset,
            )
            if not page:
                exhausted = True
                break
            offset += len(page)
            for message in page:
                block = blocks.get(message.run_id)
                if block is None:
                    blocks[message.run_id] = [message]
                    block_order.append(message.run_id)
                else:
                    block.append(message)
                    # The scan walks newest → oldest, so this message sits
                    # before every message already seen for the run: the
                    # run started earlier than the scan thought, and its
                    # block belongs after the newer-starting ones.
                    block_order.remove(message.run_id)
                    block_order.append(message.run_id)
            selected, final = _select(
                [blocks[run_id] for run_id in block_order],
                last_resolved=False,
            )
            if final:
                break
        if exhausted:
            # The oldest fetched turn is complete — every message exists.
            selected, _ = _select(
                [blocks[run_id] for run_id in block_order],
                last_resolved=True,
            )

        ordered: list[Any] = []
        for turn in reversed(selected):
            ordered.extend(reversed(turn))
        return [
            {"role": message.role, "content": message.content} for message in ordered
        ]

    # -- snapshot -------------------------------------------------------

    def get_snapshot(self, run_id: str) -> RunSnapshot | None:
        """Assemble the spec AC3 snapshot.

        Returns None when the run_id is unknown. The snapshot includes:

        * ``status`` / ``terminal_type`` — run lifecycle state
        * ``sequence_cursor`` — last persisted event sequence
        * ``messages`` — conversation history (user + assistant)
        * ``pending_actions`` — tool_calls without matching tool_result
        """
        run = self.control.get_run(run_id)
        if run is None:
            return None

        messages = self.checkpoints.get_messages(
            conversation_id=run.conversation_id
        )
        sequence_cursor = self.checkpoints.get_last_sequence(run_id)
        pending = self.checkpoints.get_pending_actions(run_id)

        return RunSnapshot(
            run_id=run.run_id,
            request_id=run.request_id,
            conversation_id=run.conversation_id,
            profile_id=run.profile_id,
            status=run.status,
            terminal_type=run.terminal_type,
            created_at=run.created_at,
            updated_at=run.updated_at,
            terminal_at=run.terminal_at,
            sequence_cursor=sequence_cursor,
            messages=[
                {
                    "role": m.role,
                    "content": m.content,
                    "sequence": m.sequence,
                    "run_id": m.run_id,
                    "created_at": m.created_at,
                }
                for m in messages
            ],
            pending_actions=pending,
        )

    # -- cancel ---------------------------------------------------------

    def request_cancel(
        self,
        *,
        run_id: str,
        idempotency_key: str,
    ) -> CancelOutcome:
        """Idempotent cancel — spec AC5.

        The idempotency_key is required so duplicate HTTP cancel
        requests (e.g. user double-clicks) return the original outcome
        rather than re-attempting the state transition.
        """
        accepted, reason = self.control.request_cancel(
            run_id=run_id,
            idempotency_key=idempotency_key,
        )
        return CancelOutcome(
            accepted=accepted,
            reason=reason,
            run_id=run_id,
        )

    def is_cancelled(self, run_id: str) -> bool:
        """Check whether a run has been cancelled.

        The streaming endpoint polls this between yields so a cancelled
        run stops emitting new events promptly.
        """
        run = self.control.get_run(run_id)
        if run is None:
            return False
        return run.status == "cancelled"


__all__ = [
    "ApprovalRecord",
    "ArtifactRecord",
    "CancelOutcome",
    "RunAlreadyTerminal",
    "RunNotFound",
    "RunRepository",
    "RunSnapshot",
]
