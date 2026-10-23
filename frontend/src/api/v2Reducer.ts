/**
 * Version 2 event reducer — pure function that maps v2 SSE events to stable
 * UI state. Spec AC2: "run.started、progress.updated、assistant.delta、tool
 * 状态、heartbeat 和 terminal events 映射为稳定 UI 状态".
 *
 * The reducer is intentionally decoupled from React. SuperAgent.tsx feeds
 * events through `reduceV2Event` and applies the resulting patch to its
 * own state. This keeps the v2 contract testable without rendering.
 *
 * Spec decision #27 maps conceptual labels to wire types:
 *   - run.started     → ``run_started`` (distinct event, first in stream)
 *   - progress.updated → ``progress``
 *   - assistant.delta → ``text``
 *   - tool status     → ``tool_call`` / ``tool_result``
 *   - heartbeat       → ``heartbeat`` (keepalive, no state change)
 *   - terminal events → ``done`` / ``error`` / ``cancelled``
 *
 * Ticket 11 adds the durable-approval lifecycle to the public contract:
 *   - approval.required → set ``pendingApproval`` (UI shows approve/reject)
 *   - run.paused        → set status=``paused`` (stream ends; run alive)
 *   - approval.resolved → clear ``pendingApproval`` + record decision
 *   - (resume stream)   → ``running`` again via ``run_started`` or
 *                          subsequent ``progress``/``text`` events
 *
 * ``paused`` is NOT a terminal state: the reducer's sticky guard only
 * locks ``done``/``error``/``cancelled`` so the ``paused`` → ``running``
 * transition on resume remains legal.
 */

import type {
  V2EventEnvelope,
  V2EventType,
  V2ApprovalRequest,
  V2MemoryCandidate,
  V2Artifact,
  V2RunSnapshot,
} from './chatV2'
import { TERMINAL_V2_EVENT_TYPES, STREAM_TERMINAL_V2_EVENT_TYPES, V2_TERMINAL_RUN_STATUSES } from './chatV2'

export interface V2ToolTrace {
  type: 'tool_call' | 'tool_result' | 'artifact_progress'
  tool: string
  server_id?: string
  status?: string
  content?: string
  args?: Record<string, unknown>
  sequence: number
}

export type V2RunStatus =
  | 'idle'
  | 'running'
  | 'paused' // Ticket 11: awaiting user approval; stream closed, run alive
  | 'done'
  | 'error'
  | 'cancelled'
  // BCDEF ticket 11: first-class client run status for runs whose
  // external effect outcome is unknown. Terminal and NON-continuable —
  // distinct from a generic ``error`` (the UI shows "awaiting background
  // reconciliation" and never offers resume/auto-retry), and repairing it
  // is ledger reconciliation, not a model call.
  | 'outcome_unknown'

// Run statuses after which the sticky guard locks the reducer — no
// further event may resurrect the run (spec: done / error / cancelled /
// outcome_unknown must not continue execution).
const STICKY_TERMINAL_STATUSES: ReadonlySet<V2RunStatus> = new Set<V2RunStatus>([
  'done',
  'error',
  'cancelled',
  'outcome_unknown',
])

/** One observed progress-sequence hole (BCDEF ticket 11). */
export interface V2ProgressGap {
  from_sequence: number
  to_sequence: number
  dropped: number
}

export interface V2ConversationState {
  /** Accumulated assistant text deltas (concatenated in arrival order). */
  assistantText: string
  /** Human-readable progress message from the latest `progress` event. */
  progressMessage: string
  /** Tool trace events for the lifecycle panel. */
  toolTrace: V2ToolTrace[]
  /** Highest sequence number observed. Used as the reconnect cursor. */
  sequenceCursor: number
  /** Current run status. Terminal states are sticky. */
  status: V2RunStatus
  /** Terminal error code/message when status === 'error'. */
  errorMessage: string
  /** True after the first non-terminal event arrives (used to toggle UI). */
  started: boolean
  /** Number of unknown/invalid frames dropped — surfaced for monitoring. */
  droppedCount: number
  /**
   * Ticket 11: pending approval awaiting user decision. Set by
   * ``approval.required``, cleared by ``approval.resolved``. The UI
   * renders the approve/reject buttons while this is non-null. Persisted
   * across refresh via V2PersistedRun so recovery can re-show the prompt.
   */
  pendingApproval: V2ApprovalRequest | null
  /**
   * Ticket 11: the most recent decision for the current approval, recorded
   * by ``approval.resolved``. Used by the UI to show "处理中" / "已解决"
   * states. Cleared when a new ``approval.required`` arrives.
   */
  lastApprovalDecision: 'approved' | 'rejected' | null
  /**
   * Ticket 11: human-readable reason when a resume attempt fails (e.g.
   * "审批已过期" / "沙箱租赁已丢失"). Set by the UI layer when
   * ``submitV2Approval`` returns a 409; cleared on the next
   * ``approval.required`` event. The reducer itself is network-free —
   * this field is a UI-driven patch surface so the rest of the state
   * stays pure w.r.t. wire events.
   */
  resumeErrorMessage: string
  pendingMemoryConsent: V2MemoryCandidate | null
  /**
   * A02: artifacts delivered during the current run via
   * ``artifact.available`` (in arrival order). The UI assigns these to
   * the active assistant message; recovery rebuilds the message from
   * here. Cleared never mid-run — a run can deliver multiple artifacts.
   */
  artifacts: V2Artifact[]
  /**
   * A02: human-review notices from ``effect.outcome_unknown`` — the
   * server emits these before the terminal error so the client can show
   * "需要人工核查" context beyond the bare error text. BCDEF ticket 11
   * reuses this surface for ``progress.gap`` notices ("dropped progress
   * is visible, never mistaken for completion").
   */
  reviewNotices: string[]
  /**
   * BCDEF ticket 11: progress-sequence holes observed from
   * ``progress.gap`` events (server-replayed) or live stream jumps
   * (client-synthesized). ``droppedCount`` aggregates their totals.
   */
  progressGaps: V2ProgressGap[]
  /**
   * BCDEF ticket 11: how many stream-terminal frames arrived AFTER the
   * run already reached its one legitimate terminal. Contract says one
   * terminal per run; duplicates are ignored for rendering and counted
   * here so the anomaly stays observable.
   */
  duplicateTerminalCount: number
}

export const initialV2State: V2ConversationState = {
  assistantText: '',
  progressMessage: '',
  toolTrace: [],
  sequenceCursor: 0,
  status: 'idle',
  errorMessage: '',
  started: false,
  droppedCount: 0,
  pendingApproval: null,
  lastApprovalDecision: null,
  resumeErrorMessage: '',
  pendingMemoryConsent: null,
  artifacts: [],
  reviewNotices: [],
  progressGaps: [],
  duplicateTerminalCount: 0,
}

/**
 * Apply a v2 event to the conversation state. Returns a new state object;
 * the input is not mutated. Terminal states are sticky: once the run
 * reaches `done`/`error`/`cancelled`, subsequent events are ignored
 * (the stream connection also stops, but the guard defends in depth).
 *
 * Ticket 11: ``paused`` is intentionally NOT in the sticky set so the
 * resume stream can transition ``paused`` → ``running``.
 */
export function reduceV2Event(
  state: V2ConversationState,
  event: V2EventEnvelope,
): V2ConversationState {
  // Terminal states are sticky. The server should not emit after a terminal,
  // but the reducer defends in depth so a buggy server cannot resurrect a
  // dead run in the UI. ``paused`` is deliberately excluded so the resume
  // stream can move the run back to ``running``.
  // BCDEF ticket 11: ``outcome_unknown`` joins the sticky set (terminal and
  // NON-continuable), and a stream-terminal frame arriving after the run's
  // one legitimate terminal is COUNTED — duplicates are ignored for
  // rendering but never silently invisible (spec: at most one terminal
  // rendered per run; duplicates from replay are ignored and counted).
  if (STICKY_TERMINAL_STATUSES.has(state.status)) {
    if (STREAM_TERMINAL_V2_EVENT_TYPES.has(event.type)) {
      return { ...state, duplicateTerminalCount: state.duplicateTerminalCount + 1 }
    }
    return state
  }

  // Sequence monotonicity: ignore events with sequence <= cursor. This
  // handles duplicate frames from reconnect replay without forcing the
  // caller to dedupe.
  if (event.sequence <= state.sequenceCursor) {
    return state
  }

  const base: V2ConversationState = {
    ...state,
    sequenceCursor: event.sequence,
    started: true,
  }

  switch (event.type as V2EventType) {
    case 'run_started':
      // Spec decision #27 + AC2: run_started is a distinct event signaling
      // the run has begun. It transitions idle → running but does not
      // carry a progress message. ``started`` is set by the base object.
      // Ticket 11: a resume stream also opens with run_started (or the
      // server may skip it and go straight to progress/text); either
      // way, idle/paused → running is the correct transition.
      return {
        ...base,
        status: 'running',
      }

    case 'progress':
      return {
        ...base,
        status: 'running',
        progressMessage: readString(event.payload, 'message') || base.progressMessage,
      }

    case 'progress.gap': {
      // BCDEF ticket 11: droppable progress left a hole in the durable
      // sequence — the server's replay (or the live stream loop's local
      // detection) surfaced it as this event. Render it as a visible
      // notice + dropped count, never as silence: a gap must not be
      // mistaken for stalled progress or completion. Progress-class: it
      // does NOT change the terminal semantics of the run.
      const from = readNumber(event.payload, 'from_sequence')
      const to = readNumber(event.payload, 'to_sequence')
      const declared = readNumber(event.payload, 'dropped')
      const dropped = Math.max(declared, to > from ? to - from - 1 : 0)
      const notice =
        readString(event.payload, 'message') ||
        `部分进度更新丢失（序列 ${from} → ${to}，共 ${dropped} 条）`
      return {
        ...base,
        status: 'running',
        droppedCount: base.droppedCount + dropped,
        progressGaps: [...base.progressGaps, { from_sequence: from, to_sequence: to, dropped }],
        reviewNotices: [...base.reviewNotices, notice],
      }
    }

    case 'text':
      return {
        ...base,
        status: 'running',
        assistantText: base.assistantText + readString(event.payload, 'content'),
      }

    case 'thought':
      // Reasoning side channel: it never touches the public answer text, but
      // it does prove the run is still active.
      return {
        ...base,
        status: 'running',
      }

    case 'heartbeat':
      // Spec decision #27 + AC2: heartbeat is a keepalive event. It
      // advances the sequence cursor and marks the run as started but
      // does NOT change status, progress, or text. This keeps the
      // connection alive without flickering the UI.
      return base

    case 'tool_call':
      return {
        ...base,
        status: 'running',
        toolTrace: [
          ...base.toolTrace,
          {
            type: 'tool_call',
            tool: readString(event.payload, 'tool'),
            server_id: readString(event.payload, 'server_id'),
            args: readObject(event.payload, 'args'),
            sequence: event.sequence,
          },
        ],
      }

    case 'tool_result':
      return {
        ...base,
        status: 'running',
        toolTrace: [
          ...base.toolTrace,
          {
            type: 'tool_result',
            tool: readString(event.payload, 'tool'),
            server_id: readString(event.payload, 'server_id'),
            status: readString(event.payload, 'status'),
            content: readString(event.payload, 'content'),
            sequence: event.sequence,
          },
        ],
      }

    case 'artifact_candidate':
      // A02: the sandbox has produced a candidate artifact and host-side
      // export is pending. Surface as a trace line; the download card
      // only appears on artifact.available.
      return {
        ...base,
        status: 'running',
        toolTrace: [
          ...base.toolTrace,
          {
            type: 'artifact_progress',
            tool: readString(event.payload, 'tool') || 'artifact',
            content:
              readString(event.payload, 'message') || '正在生成产物…',
            sequence: event.sequence,
          },
        ],
      }

    case 'artifact.available':
      // A02: collect the verified artifact (payload carries ONLY opaque
      // artifact_id + public metadata per spec AC3). The UI renders a
      // download card pointing at /api/v2/artifacts/{artifact_id}.
      {
        const artifact = readArtifact(event.payload)
        return artifact
          ? { ...base, status: 'running', artifacts: [...base.artifacts, artifact] }
          : { ...base, status: 'running', droppedCount: base.droppedCount + 1 }
      }

    case 'effect.outcome_unknown':
      // A02: a side effect finished with an unknown outcome — the server
      // contract emits this BEFORE the terminal error so the client can
      // show the human-review context, not just the bare error text.
      return {
        ...base,
        status: 'running',
        reviewNotices: [
          ...base.reviewNotices,
          readString(event.payload, 'message') ||
            readString(event.payload, 'code') ||
            '部分操作结果未知，需要人工核查',
        ],
      }

    case 'approval.required':
      // Ticket 11 spec AC2 + AC6: surface the approval to the UI. The
      // server emits this BEFORE ``run.paused`` so the client has the
      // approval payload in hand when the stream closes. Reset any
      // stale decision/resume error so the prompt renders cleanly.
      return {
        ...base,
        pendingApproval: readApprovalRequest(event.payload),
        lastApprovalDecision: null,
        resumeErrorMessage: '',
      }

    case 'approval.resolved':
      // Ticket 11 spec AC4: the decision has been recorded server-side
      // and the run is resuming. Clear the pending prompt so the UI
      // dismisses the approve/reject buttons. ``lastApprovalDecision``
      // lets the UI show a brief "已批准" / "已拒绝" confirmation before
      // the next events arrive.
      return {
        ...base,
        status: 'running',
        pendingApproval: null,
        lastApprovalDecision: readApprovalDecision(event.payload),
        resumeErrorMessage: '',
      }

    case 'memory.consent_required':
      return {
        ...base,
        pendingMemoryConsent: readMemoryCandidate(event.payload),
      }

    case 'run.paused':
      // Ticket 11 spec AC3: the graph has persistently interrupted and
      // the SSE stream is about to close. Transition to ``paused`` so
      // the UI knows the run is awaiting approval (not dead). The
      // stream connection's onTerminal handler also fires; the caller
      // must NOT clear ``isSending`` or ``v2RunRef`` on this event
      // (only on done/error/cancelled).
      return {
        ...base,
        status: 'paused',
      }

    case 'done':
      return {
        ...base,
        status: 'done',
        progressMessage: '',
        pendingApproval: null,
      }

    case 'error': {
      // BCDEF ticket 11: an error terminal whose stable code marks an
      // unknown external-effect outcome maps onto the first-class
      // ``outcome_unknown`` run status — rendered apart from a generic
      // error and never continuable (no resume affordance, no automatic
      // model call; repairing it is ledger reconciliation).
      const code = readString(event.payload, 'code')
      const unknownOutcome =
        code === 'effect_outcome_unknown' ||
        code === 'outcome_unknown' ||
        readString(event.payload, 'run_status') === 'outcome_unknown'
      return {
        ...base,
        status: unknownOutcome ? 'outcome_unknown' : 'error',
        errorMessage:
          readString(event.payload, 'message') ||
          code ||
          (unknownOutcome ? '操作结果未知，等待后台对账修复' : '运行出错'),
        progressMessage: '',
        pendingApproval: null,
      }
    }

    case 'cancelled':
      return {
        ...base,
        status: 'cancelled',
        progressMessage: '',
        pendingApproval: null,
      }

    default:
      // Unknown event type — should not reach here because the parser
      // drops unknown types. Counted as dropped for monitoring.
      return {
        ...base,
        droppedCount: base.droppedCount + 1,
      }
  }
}

function readMemoryCandidate(payload: Record<string, unknown>): V2MemoryCandidate | null {
  const candidate_id = readString(payload, 'candidate_id')
  const category = readString(payload, 'category')
  const sensitivity = readString(payload, 'sensitivity')
  const preview = readString(payload, 'preview')
  const expires_at = readString(payload, 'expires_at')
  if (!candidate_id || !category || !sensitivity || !preview || !expires_at) return null
  return { candidate_id, category, sensitivity, preview, expires_at }
}

/**
 * Apply a batch of events (e.g. replayed from /events on reconnect).
 * Events are applied in sequence order; out-of-order events are ignored
 * by the reducer's monotonicity guard.
 */
export function reduceV2Events(
  state: V2ConversationState,
  events: V2EventEnvelope[],
): V2ConversationState {
  const sorted = [...events].sort((a, b) => a.sequence - b.sequence)
  return sorted.reduce(reduceV2Event, state)
}

/** True when the run has reached a terminal state. */
export function isV2Terminal(state: V2ConversationState): boolean {
  return STICKY_TERMINAL_STATUSES.has(state.status)
}

/**
 * True when the run is terminal AND continuable-free: done, error and
 * outcome_unknown must never offer or trigger continued execution —
 * only ``paused`` may resume and ``cancelled`` is a replayable terminal
 * that likewise never re-executes (BCDEF ticket 11).
 */
export function isV2NonContinuable(state: V2ConversationState): boolean {
  return STICKY_TERMINAL_STATUSES.has(state.status)
}

/** True when the run is paused awaiting user approval (Ticket 11). */
export function isV2Paused(state: V2ConversationState): boolean {
  return state.status === 'paused'
}

/**
 * BCDEF ticket 11 — authoritative snapshot reconciliation.
 *
 * The run snapshot (plus the durable replay) is THE recovery source
 * after stream close, refresh, cancel and resume. This pure step folds a
 * fetched ``/snapshot`` into the replayed reducer state:
 *
 * * a terminal-looking snapshot that never became durable
 *   (``terminal_persist_failed`` — or the server's fallback of the same
 *   name for a terminal control status with no persisted event) renders
 *   the stable error, never ``done``;
 * * ``outcome_unknown`` maps onto the first-class client status (distinct
 *   from a generic error, non-continuable, no automatic model call);
 * * a terminal snapshot NEVER leaves the UI "streaming" — even when the
 *   replay contains no terminal frame, the snapshot finalizes the status;
 * * ``paused`` stays non-terminal, rehydrates the pending approval from
 *   ``pending_approvals`` when the replay could not deliver it;
 * * a replayed run terminal wins over the snapshot status (it is the more
 *   specific durable fact), except that a replayed ``done`` contradicted
 *   by ``terminal_persist_failed`` is downgraded to the stable error.
 */
export function reconcileV2Snapshot(
  state: V2ConversationState,
  snapshot: V2RunSnapshot | null,
): V2ConversationState {
  if (!snapshot) return state

  const terminalPersistFailed = snapshot.terminal_persist_failed === true
  const snapshotStatus = snapshot.status
  const snapshotTerminal = V2_TERMINAL_RUN_STATUSES.has(snapshotStatus)

  if (snapshotTerminal) {
    // A replayed sticky terminal already finalized the run.
    if (STICKY_TERMINAL_STATUSES.has(state.status)) {
      // …except the AC7 contradiction: never keep a ``done`` the server
      // cannot prove is durable.
      if (state.status === 'done' && terminalPersistFailed) {
        return {
          ...state,
          status: 'error',
          errorMessage: snapshot.error_code || 'terminal_persist_failed',
        }
      }
      return state
    }
    if (terminalPersistFailed) {
      return {
        ...state,
        status: 'error',
        errorMessage: snapshot.error_code || 'terminal_persist_failed',
        progressMessage: '',
        pendingApproval: null,
        started: true,
      }
    }
    if (snapshotStatus === 'outcome_unknown') {
      return {
        ...state,
        status: 'outcome_unknown',
        errorMessage: '操作结果未知（outcome_unknown），等待后台对账修复，不会自动重放。',
        progressMessage: '',
        pendingApproval: null,
        started: true,
      }
    }
    if (snapshotStatus === 'admission_failed') {
      return {
        ...state,
        status: 'error',
        errorMessage: snapshot.error_code || 'admission_failed',
        progressMessage: '',
        pendingApproval: null,
        started: true,
      }
    }
    if (snapshotStatus === 'done') {
      return {
        ...state,
        status: 'done',
        progressMessage: '',
        pendingApproval: null,
        started: true,
      }
    }
    // error / cancelled — mirror the server terminal.
    return {
      ...state,
      status: snapshotStatus === 'cancelled' ? 'cancelled' : 'error',
      errorMessage:
        snapshotStatus === 'error'
          ? snapshot.terminal_error_code || snapshot.error_code || '运行出错'
          : state.errorMessage,
      progressMessage: '',
      pendingApproval: null,
      started: true,
    }
  }

  if (snapshotStatus === 'paused') {
    // Non-terminal: the run awaits approval. Do NOT resurrect a sticky
    // local terminal (the server status could lag a durably emitted one).
    if (STICKY_TERMINAL_STATUSES.has(state.status)) return state
    const approval =
      state.pendingApproval ?? readSnapshotApproval(snapshot) ?? null
    return {
      ...state,
      status: 'paused',
      pendingApproval: approval,
      started: true,
    }
  }

  // Snapshot still ``running``/``admitted``: nothing to add — the caller
  // keeps polling; never force a terminal from a non-terminal snapshot.
  return state
}

/** Map snapshot ``pending_approvals`` into the ``approval.required`` shape. */
function readSnapshotApproval(snapshot: V2RunSnapshot): V2ApprovalRequest | null {
  const raw = snapshot.pending_approvals?.[0]
  if (!raw || typeof raw.approval_id !== 'string' || typeof raw.tool !== 'string') {
    return null
  }
  return {
    approval_id: raw.approval_id,
    tool: raw.tool,
    args: (raw.args ?? {}) as Record<string, unknown>,
    risk_level:
      raw.risk_level === 'low' || raw.risk_level === 'medium' || raw.risk_level === 'high'
        ? raw.risk_level
        : 'high',
    impact: typeof raw.impact === 'string' ? raw.impact : '',
    expires_at: typeof raw.expires_at === 'string' ? raw.expires_at : '',
    message: typeof raw.message === 'string' ? raw.message : undefined,
  }
}

/**
 * BCDEF ticket 11: the assistant text authoritative in the snapshot for
 * this run — the fallback when the event replay cannot complete the
 * picture (e.g. the assistant frame was written to the conversation DB
 * but the SSE event row was lost).
 */
export function assistantTextFromSnapshot(
  snapshot: V2RunSnapshot,
  runId?: string,
): string {
  const candidates = snapshot.messages.filter(m => m.role === 'assistant')
  const scoped = runId ? candidates.filter(m => !m.run_id || m.run_id === runId) : candidates
  const last = scoped[scoped.length - 1]
  return last ? last.content : ''
}

/** Map a v2 status to a user-visible Chinese label. */
export function v2StatusLabel(state: V2ConversationState): string {
  switch (state.status) {
    case 'idle':
      return '等待开始'
    case 'running':
      return state.progressMessage || '正在思考'
    case 'paused':
      // Ticket 11: prefer the approval message if available, then a
      // generic "等待批准" fallback. The UI renders the actual
      // approve/reject buttons separately based on ``pendingApproval``.
      return state.pendingApproval?.message || '等待您的批准'
    case 'done':
      return '已完成'
    case 'error':
      return state.errorMessage || '运行出错'
    case 'outcome_unknown':
      // BCDEF ticket 11: rendered distinctly from a generic error — the
      // user is told the outcome is unknown and awaiting background
      // reconciliation, not that the run "failed with a retryable error".
      return state.errorMessage || '结果未知，等待后台对账修复（outcome_unknown）'
    case 'cancelled':
      return '已取消'
    default:
      return ''
  }
}

/** Helper: read a string field from the payload, undefined → ''. */
function readString(payload: Record<string, unknown>, key: string): string {
  const value = payload[key]
  return typeof value === 'string' ? value : ''
}

/** Helper: read a numeric field from the payload, undefined/NaN → 0. */
function readNumber(payload: Record<string, unknown>, key: string): number {
  const value = payload[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

/** Helper: read an object field, undefined → undefined. */
function readObject(
  payload: Record<string, unknown>,
  key: string,
): Record<string, unknown> | undefined {
  const value = payload[key]
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>
  }
  return undefined
}

/**
 * Helper: read the ``artifact.available`` payload into a typed
 * ``V2Artifact``. Returns null unless the two identity fields are
 * present — a malformed artifact payload must not crash the reducer.
 * Exported for the UI layer, which attaches artifacts to messages.
 */
export function readArtifact(payload: Record<string, unknown>): V2Artifact | null {
  const artifactId = payload['artifact_id']
  const filename = payload['public_filename']
  if (typeof artifactId !== 'string' || !artifactId) return null
  if (typeof filename !== 'string' || !filename) return null
  const size = payload['size_bytes']
  return {
    artifact_id: artifactId,
    public_filename: filename,
    mime_type: readString(payload, 'mime_type'),
    size_bytes: typeof size === 'number' ? size : 0,
    sha256: typeof payload['sha256'] === 'string' ? payload['sha256'] : undefined,
    expires_at: typeof payload['expires_at'] === 'string' ? payload['expires_at'] : undefined,
    message: typeof payload['message'] === 'string' ? payload['message'] : undefined,
  }
}

/**
 * Helper: read the ``approval.required`` payload into a typed
 * ``V2ApprovalRequest``. Defensive against missing/invalid fields so a
 * buggy server cannot crash the reducer.
 */
function readApprovalRequest(payload: Record<string, unknown>): V2ApprovalRequest | null {
  const approvalId = payload['approval_id']
  const tool = payload['tool']
  if (typeof approvalId !== 'string' || typeof tool !== 'string') {
    return null
  }
  const args = payload['args']
  const riskLevel = payload['risk_level']
  const validRisks: Array<V2ApprovalRequest['risk_level']> = ['low', 'medium', 'high']
  const risk = typeof riskLevel === 'string' && (validRisks as string[]).includes(riskLevel)
    ? (riskLevel as V2ApprovalRequest['risk_level'])
    : 'high' // fail safe: unknown risk defaults to high
  return {
    approval_id: approvalId,
    tool,
    args: (args && typeof args === 'object' && !Array.isArray(args)
      ? args
      : {}) as Record<string, unknown>,
    risk_level: risk,
    impact: readString(payload, 'impact'),
    expires_at: readString(payload, 'expires_at'),
    message: typeof payload['message'] === 'string' ? payload['message'] : undefined,
  }
}

/** Helper: read the ``approval.resolved`` decision. */
function readApprovalDecision(
  payload: Record<string, unknown>,
): 'approved' | 'rejected' | null {
  const status = payload['status']
  const decision = payload['decision']
  if (status === 'approved' || decision === 'approve') return 'approved'
  if (status === 'rejected' || decision === 'reject') return 'rejected'
  return null
}

// Re-export for callers that need the terminal set.
export { TERMINAL_V2_EVENT_TYPES }
