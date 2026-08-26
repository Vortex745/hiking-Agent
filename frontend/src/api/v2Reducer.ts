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
} from './chatV2'
import { TERMINAL_V2_EVENT_TYPES } from './chatV2'

export interface V2ToolTrace {
  type: 'tool_call' | 'tool_result'
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
  if (state.status === 'done' || state.status === 'error' || state.status === 'cancelled') {
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

    case 'text':
      return {
        ...base,
        status: 'running',
        assistantText: base.assistantText + readString(event.payload, 'content'),
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

    case 'error':
      return {
        ...base,
        status: 'error',
        errorMessage:
          readString(event.payload, 'message') ||
          readString(event.payload, 'code') ||
          '运行出错',
        pendingApproval: null,
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
  return (
    state.status === 'done' ||
    state.status === 'error' ||
    state.status === 'cancelled'
  )
}

/** True when the run is paused awaiting user approval (Ticket 11). */
export function isV2Paused(state: V2ConversationState): boolean {
  return state.status === 'paused'
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
