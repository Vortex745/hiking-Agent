"""Versioned product event envelope for Chat v2.

The envelope is the only event shape the frontend and Gateway may consume.
Raw LangChain / UniHarness events are never transmitted; the SSE adapter
maps them to this envelope or drops them (counted as ``unknown_dropped``).

Invariants enforced by :class:`EventStream`:

* ``version`` is always ``"2"``.
* ``sequence`` is strictly monotonic starting at 1 within a single run.
* Exactly one terminal event (``error`` or ``done``) is emitted per stream.
* Transport closure does not substitute for a terminal event; callers
  that observe EOF without a terminal must treat the run as ``outcome_unknown``.

The envelope is a frozen dataclass so downstream code cannot mutate
sequence or terminal state. Serialization is JSON with ``ensure_ascii=False``
so Chinese content stays readable in logs.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


EventVersion = Literal["2"]
# Spec decision #27: public events are limited to run start, user-safe
# progress, assistant delta, tool start/completion, approval required/resolved,
# artifact available, memory consent required, pause, terminal outcomes, and
# heartbeat. ``run_started`` and ``heartbeat`` are added here so the parser
# and reducer recognize them; approval/memory/pause types arrive with later
# tickets. ``artifact_candidate`` (Ticket 09) signals a sandbox produced a
# candidate artifact pending host-side export. ``artifact.available``
# (Ticket 10) signals the host has verified the manifest and persisted the
# artifact — the payload carries only the opaque ``artifact_id`` and
# public metadata, never sandbox or host paths (spec AC3).
#
# Ticket 11 adds the approval + pause event types:
# * ``approval.required`` — the PermissionGate classified the tool as
#   high-risk; the user must approve or reject before the tool executes.
#   Payload carries tool name, risk level, impact summary, approval id,
#   and expiry timestamp (spec AC2).
# * ``approval.resolved`` — the user decided (approve/reject) or the
#   approval expired. Payload carries the decision and the resulting
#   run state (resume vs. terminate).
# * ``run.paused`` — the stream is ending because the run is paused
#   awaiting approval. The run is NOT terminal; a subsequent
#   ``POST /approvals/{approval_id}/decision`` resumes it. This is a
#   "stream terminal" but not a "run terminal" — the EventStream treats
#   it as terminal so exactly-one-terminal-per-stream still holds, but
#   the run's DB status transitions to ``paused`` rather than
#   ``done``/``error``/``cancelled``.
#
# Ticket 12 adds the effect ledger event type:
# * ``effect.outcome_unknown`` — a side-effecting tool call crashed in
#   the ``executing`` state and the result cannot be safely determined.
#   Emitted BEFORE the ``error`` stream terminal so clients can surface
#   a clear "needs human review" message. The run terminates with
#   ``error`` (NOT auto-retried, spec AC3); a new run + approval is
#   required to retry. This event is NOT itself a stream terminal —
#   the ``error`` terminal still follows to satisfy the
#   exactly-one-terminal-per-stream contract.
# ``thought`` carries model reasoning the public text filter removed
# (``<think>`` blocks or a provider reasoning delta). It is optional: a
# provider that emits no reasoning simply produces no such event.
# ``progress.gap`` (BCDEF ticket 01) is a progress-class projection emitted
# when one or more droppable progress events were not persisted. The payload
# carries ``from_sequence``, ``to_sequence`` and ``dropped`` so replay can
# surface the hole instead of treating it as completion. It is additive: the
# envelope shape and the terminal sets are unchanged.
EventType = Literal[
    "run_started",
    "progress",
    "progress.gap",
    "text",
    "thought",
    "tool_call",
    "tool_result",
    "artifact_candidate",
    "artifact.available",
    "approval.required",
    "approval.resolved",
    "memory.consent_required",
    "run.paused",
    "effect.outcome_unknown",
    "error",
    "done",
    "cancelled",
    "heartbeat",
]

# Terminal events signal end-of-stream. Exactly one must be emitted per run.
# Per spec decision #28: every stream ends with exactly one completed, failed,
# cancelled, or paused event. ``cancelled`` is emitted by the streaming
# endpoint when it observes the run has been cancelled via the cancel API.
# ``run.paused`` (Ticket 11) ends the current stream but the run itself
# is NOT terminal — it transitions to ``paused`` DB status and waits for
# the decision endpoint to resume it. The EventStream treats ``run.paused``
# as a stream-terminal so the wire contract (exactly one stream terminal)
# is preserved, but :func:`is_terminal` returns False for ``run.paused``
# because the run's lifecycle is not over.
TERMINAL_EVENT_TYPES: frozenset[EventType] = frozenset(
    {"error", "done", "cancelled", "run.paused"}
)

# Stream-terminal events that do NOT end the run's lifecycle — the run
# may be resumed later. ``run.paused`` is the only member today; future
# interrupt types (e.g. memory consent) would join this set.
RUN_LIFECYCLE_NON_TERMINAL: frozenset[EventType] = frozenset({"run.paused"})


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    """A single product event in the v2 stream.

    Field order matches the wire format. ``payload`` is open-ended but the
    adapter must never put raw framework events there — only mapped,
    user-safe content.

    Spec decision #26: the SSE envelope has fixed version, event id,
    sequence, run id, conversation id, timestamp, type, and data.
    ``event_id`` is a server-generated UUID per event; ``timestamp`` is
    UTC ISO 8601. Both are populated by :meth:`EventStream.build` so the
    wire envelope and the persisted row share the same values.

    ``run_id`` is the server-generated id of the run that owns this event
    (spec decision #26). It is empty only when persistence is off — in
    that case clients cannot call ``/snapshot`` / ``/cancel`` / ``/events``
    and must rely on the in-memory stream alone.
    """

    version: EventVersion
    sequence: int
    type: EventType
    payload: dict[str, Any]
    request_id: str = ""
    conversation_id: str = ""
    run_id: str = ""
    event_id: str = ""
    timestamp: str = ""

    def to_json_line(self) -> str:
        """Serialize to a single SSE ``data:`` payload line (no trailing newline)."""
        return json.dumps(
            {
                "version": self.version,
                "event_id": self.event_id,
                "sequence": self.sequence,
                "run_id": self.run_id,
                "conversation_id": self.conversation_id,
                "timestamp": self.timestamp,
                "type": self.type,
                "request_id": self.request_id,
                "payload": self.payload,
            },
            ensure_ascii=False,
        )


class TerminalEventAlreadyEmitted(RuntimeError):
    """Raised when a stream tries to emit after a terminal event.

    This is a programming error: terminal events close the stream. The
    guard exists so adapters cannot accidentally produce two ``done``
    frames when an exception overlaps with normal completion.
    """


@dataclass(slots=True)
class EventStream:
    """Stateful helper that enforces v2 stream invariants.

    The stream tracks the next sequence number, counts dropped unknown
    events, and guarantees exactly one terminal event. It does NOT own
    I/O — callers consume ``build()`` results and push them to the wire.

    ``unknown_dropped`` is exposed in the terminal ``done`` payload so
    monitoring can detect adapter drift (e.g. UniHarness adds a new
    internal event type that the adapter doesn't map yet).

    ``run_id`` is set by the caller after :meth:`create_run_idempotent`
    returns the server-generated id. It is propagated to every built
    envelope so the frontend can call ``/snapshot/{run_id}`` etc.
    without needing a separate side channel.
    """

    request_id: str
    conversation_id: str = ""
    run_id: str = ""
    _next_sequence: int = field(default=1, init=False)
    _terminal_emitted: bool = field(default=False, init=False)
    _terminal_type: EventType | None = field(default=None, init=False)
    unknown_dropped: int = field(default=0, init=False)

    def build(self, event_type: EventType, payload: dict[str, Any]) -> EventEnvelope:
        """Build the next envelope, enforcing monotonic sequence + terminal guard.

        Raises :class:`TerminalEventAlreadyEmitted` if called again after a
        terminal event. The caller decides how to convert that to a wire
        error (typically log + close).
        """
        if self._terminal_emitted:
            raise TerminalEventAlreadyEmitted(
                f"Stream already closed by terminal event {self._terminal_type!r}; "
                f"cannot emit {event_type!r}"
            )

        # Copy payload so the stream never mutates the caller's dict. The
        # ``done`` and ``error`` envelopes attach ``unknown_dropped`` for
        # monitoring; doing this on a copy keeps the adapter's input
        # immutable. Both terminal types carry the count so adapter drift
        # is observable regardless of whether the run succeeded or failed.
        effective_payload = dict(payload)
        if event_type in TERMINAL_EVENT_TYPES and "unknown_dropped" not in effective_payload:
            effective_payload["unknown_dropped"] = self.unknown_dropped

        envelope = EventEnvelope(
            version="2",
            sequence=self._next_sequence,
            type=event_type,
            payload=effective_payload,
            request_id=self.request_id,
            conversation_id=self.conversation_id,
            run_id=self.run_id,
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._next_sequence += 1

        if event_type in TERMINAL_EVENT_TYPES:
            self._terminal_emitted = True
            self._terminal_type = event_type

        return envelope

    def drop_unknown(self, internal_type: str) -> None:
        """Record an unknown internal event that the adapter chose not to map.

        Does NOT emit an envelope. The count surfaces in the ``done`` payload
        so monitoring can detect drift without leaking internal event names
        to the user-visible stream.
        """
        self.unknown_dropped += 1

    def advance_to(self, sequence: int) -> None:
        """Advance the internal sequence counter past ``sequence``.

        Used on reconnect: stored events are replayed with their original
        sequences (bypassing :meth:`build`), so the counter must be
        advanced before synthesizing a terminal event — otherwise the
        synthesized terminal would reuse an already-emitted sequence,
        breaking monotonicity (spec AC4).
        """
        if sequence + 1 > self._next_sequence:
            self._next_sequence = sequence + 1

    def reopen_after_failed_persistence(self) -> None:
        """Re-open the terminal slot for a terminal that never left the server.

        BCDEF ticket 03: a terminal envelope is built BEFORE the assistant
        + terminal durability fence runs. When that fence fails, the frame
        was never published (the stream closes or replaces it), so one
        replacement terminal — e.g. ``assistant_persist_failed`` — must be
        buildable. The consumed sequence is NOT reused: the replacement
        gets the next sequence, keeping monotonicity. This is not a
        general-purpose reset; callers use it only inside the failed-fence
        path.
        """
        self._terminal_emitted = False
        self._terminal_type = None

    @property
    def terminal_emitted(self) -> bool:
        """True after a terminal event has been built."""
        return self._terminal_emitted

    @property
    def terminal_type(self) -> EventType | None:
        """The terminal event type emitted, or ``None`` if not yet terminal."""
        return self._terminal_type


def is_terminal(envelope: EventEnvelope) -> bool:
    """True if ``envelope`` ends the run's lifecycle (``error``, ``done``, or ``cancelled``).

    ``run_started`` and ``heartbeat`` are NOT terminal — they are lifecycle
    events that keep the stream alive. ``run.paused`` (Ticket 11) is a
    stream-terminal (ends the current SSE stream) but NOT a run-lifecycle
    terminal — the run is paused, not done, and a subsequent decision
    resumes it. Callers that need "stream is over" should use
    :data:`TERMINAL_EVENT_TYPES`; callers that need "run is over" should
    use this function.
    """
    return envelope.type in {"error", "done", "cancelled"}


def is_stream_terminal(envelope: EventEnvelope) -> bool:
    """True if ``envelope`` ends the current SSE stream.

    Includes ``run.paused`` — the stream closes when the run pauses, but
    the run's lifecycle is not over. Use :func:`is_terminal` for
    run-lifecycle checks.
    """
    return envelope.type in TERMINAL_EVENT_TYPES
