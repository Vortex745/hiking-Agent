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
  V2PlanDraftState,
  V2PlanDraftStage,
  V2PlanRouteReference,
  V2PlanType,
  V2PlanMissingField,
  V2PlanArrangementFields,
  V2PlanCompletenessRules,
  V2PlanSafety,
  V2PlanEvidenceMatch,
  V2PlanFormatIssue,
  V2PlanReviewReason,
  V2PlanRuleGap,
} from './chatV2'
import {
  TERMINAL_V2_EVENT_TYPES,
  V2_PLAN_ARRANGEMENT_KEYS,
  V2_PLAN_DRAFT_STAGES,
  V2_PLAN_SAFETY_CODES,
  V2_PLAN_TYPES,
  V2_PLAN_VERDICTS,
} from './chatV2'

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
   * "需要人工核查" context beyond the bare error text.
   */
  reviewNotices: string[]
  /**
   * Ticket 01: the conversation's plan-book draft state, set by
   * ``plan.draft_state``. ``null`` means no plan flow has been opened. The
   * UI opens the dialog only from ``awaiting_input``; ``awaiting_confirmation``
   * is the ambiguous-intent prompt and must leave the dialog closed.
   */
  planDraft: V2PlanDraftState | null
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
  planDraft: null,
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

    case 'plan.draft_state':
      // Ticket 01 AC8: the server moved this conversation's plan draft to
      // a new stage. The reducer only records it — deciding whether that
      // means "open the dialog" or "ask for confirmation" is the UI's
      // reading of ``stage``, and an unreadable payload counts as dropped
      // instead of guessing.
      {
        const draft = readPlanDraftState(event.payload)
        return draft ? { ...base, status: 'running', planDraft: draft } : base
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
 * Helper: read the ``plan.draft_state`` payload (Ticket 01) into a typed
 * ``V2PlanDraftState``. Returns null when the stage is missing or outside
 * the shared vocabulary — an unknown stage must never open a dialog on a
 * guess.
 */
/**
 * Ticket 03: read a draft's route reference. ``null`` unless BOTH ids are
 * non-empty strings — a partial pointer would let the UI name a route the
 * server never issued, which is the thing spec AC4 forbids.
 */
function readRouteSelection(
  value: unknown,
): V2PlanRouteReference | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const raw = value as Record<string, unknown>
  const comparisonId = raw['comparison_id']
  const candidateId = raw['candidate_id']
  if (typeof comparisonId !== 'string' || !comparisonId) return null
  if (typeof candidateId !== 'string' || !candidateId) return null
  return { comparison_id: comparisonId, candidate_id: candidateId }
}

/**
 * Ticket 04: read the draft's declared type. Only the two known values are
 * ever honoured; anything else (a corrupted mirror, an older server, a value a
 * client invented) falls back to what the draft demonstrably is — a route
 * reference means a basis was re-read and passed the gate on the turn that
 * wrote this payload, no reference means there is nothing to verify against.
 * The fallback can never upgrade a draft to `route_plan` on its own, because a
 * half-read reference already came out `null` above.
 */
function readPlanType(value: unknown, reference: V2PlanRouteReference | null): V2PlanType {
  if (typeof value === 'string' && V2_PLAN_TYPES.has(value as V2PlanType)) {
    return value as V2PlanType
  }
  return reference ? 'route_plan' : 'generic_draft'
}

/**
 * Ticket 05 AC1: read the server's 缺失字段清单. Each entry must carry the key,
 * the label and the status as non-empty strings, or it is dropped — a field
 * this reader cannot name is a field the UI would have to guess a marker for,
 * and guessing whether something is 待确认 or 待核验 is the AC4 failure mode.
 * A payload from a server that predates the list is an empty list, never a
 * fabricated one.
 */
function readMissingPlanFields(value: unknown): V2PlanMissingField[] {
  if (!Array.isArray(value)) return []
  const fields: V2PlanMissingField[] = []
  for (const raw of value) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) continue
    const entry = raw as Record<string, unknown>
    const key = entry['key']
    const label = entry['label']
    const status = entry['status']
    if (typeof key !== 'string' || !key) continue
    if (typeof label !== 'string' || !label) continue
    if (typeof status !== 'string' || !status) continue
    fields.push({ key, label, status })
  }
  return fields
}

/**
 * Ticket 06 AC1/AC5: read the collected supplementary arrangements. Only the
 * eight enumerated keys are taken, and only as strings — an absent, malformed
 * or hand-edited entry yields the empty string for that field, which is the
 * server's own 待确认 state rather than a guessed value. A key outside the
 * enumeration (a smuggled distance or budget total) is dropped, not drawn.
 */
function readPlanArrangements(value: unknown): V2PlanArrangementFields {
  const record = value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
  const arrangements = {} as V2PlanArrangementFields
  for (const key of V2_PLAN_ARRANGEMENT_KEYS) {
    const entry = record[key]
    arrangements[key] = typeof entry === 'string' ? entry : ''
  }
  return arrangements
}

function readStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []
}

/**
 * Ticket 06 AC4/AC5: read the completeness verdict. An unknown verdict is
 * ``null`` — the panel then says nothing — because guessing a verdict would be
 * re-deciding a rule client-side. Reason/gap/format entries need their stable
 * code/key as a non-empty string or they are dropped, exactly like
 * ``readMissingPlanFields``: a code this reader cannot name is a code no UI
 * copy exists for.
 */
function readPlanRules(value: unknown): V2PlanCompletenessRules | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const rules = value as Record<string, unknown>
  const verdict = rules['verdict']
  if (typeof verdict !== 'string' || !V2_PLAN_VERDICTS.has(verdict as V2PlanCompletenessRules['verdict'])) {
    return null
  }
  const reasons: V2PlanReviewReason[] = []
  if (Array.isArray(rules['reasons'])) {
    for (const raw of rules['reasons']) {
      if (!raw || typeof raw !== 'object' || Array.isArray(raw)) continue
      const entry = raw as Record<string, unknown>
      if (typeof entry['code'] !== 'string' || !entry['code']) continue
      if (typeof entry['message'] !== 'string' || !entry['message']) continue
      reasons.push({ code: entry['code'], message: entry['message'] })
    }
  }
  const gaps = (key: string): V2PlanRuleGap[] => {
    const out: V2PlanRuleGap[] = []
    const rawList = Array.isArray(rules[key]) ? rules[key] as unknown[] : []
    for (const raw of rawList) {
      if (!raw || typeof raw !== 'object' || Array.isArray(raw)) continue
      const entry = raw as Record<string, unknown>
      if (typeof entry['key'] !== 'string' || !entry['key']) continue
      out.push({ key: entry['key'], label: typeof entry['label'] === 'string' ? entry['label'] : entry['key'] })
    }
    return out
  }
  const formatIssues: V2PlanFormatIssue[] = []
  const rawIssues = Array.isArray(rules['format_issues']) ? rules['format_issues'] as unknown[] : []
  for (const raw of rawIssues) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) continue
    const entry = raw as Record<string, unknown>
    if (typeof entry['key'] !== 'string' || !entry['key']) continue
    if (typeof entry['message'] !== 'string' || !entry['message']) continue
    formatIssues.push({
      key: entry['key'],
      label: typeof entry['label'] === 'string' ? entry['label'] : entry['key'],
      message: entry['message'],
    })
  }
  const matches: V2PlanEvidenceMatch[] = []
  const rawMatches = Array.isArray(rules['matches']) ? rules['matches'] as unknown[] : []
  for (const raw of rawMatches) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) continue
    const entry = raw as Record<string, unknown>
    if (typeof entry['arrangement_key'] !== 'string' || !entry['arrangement_key']) continue
    if (typeof entry['status'] !== 'string' || !entry['status']) continue
    matches.push({
      arrangement_key: entry['arrangement_key'],
      arrangement_label: typeof entry['arrangement_label'] === 'string' ? entry['arrangement_label'] : entry['arrangement_key'],
      fact_fields: readStringArray(entry['fact_fields']),
      status: entry['status'],
      source_ids: readStringArray(entry['source_ids']),
    })
  }
  return {
    verdict: verdict as V2PlanCompletenessRules['verdict'],
    reasons,
    missing_core: gaps('missing_core'),
    missing_non_core: gaps('missing_non_core'),
    format_issues: formatIssues,
    matches,
  }
}

/**
 * Ticket 07 AC2/AC5: read the gate's refusal. An unknown or malformed record is
 * `null` — the panel then says nothing about safety, because guessing a block
 * reason would be re-deciding the gate client-side (and guessing the *absence* of
 * one would silently un-block a dangerous draft). Codes outside
 * ``V2_PLAN_SAFETY_CODES`` are dropped rather than rendered as an unexplained
 * string, and the gate's own reason lines are kept only as strings.
 */
function readPlanSafety(value: unknown): V2PlanSafety | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const safety = value as Record<string, unknown>
  const rawCodes = Array.isArray(safety['codes']) ? safety['codes'] as unknown[] : []
  const codes = rawCodes.filter(
    (code): code is string => typeof code === 'string' && V2_PLAN_SAFETY_CODES.includes(code),
  )
  if (codes.length === 0) return null
  return {
    codes,
    hard_stop: safety['hard_stop'] === true,
    formal_route_plan_allowed: safety['formal_route_plan_allowed'] !== false,
    message: typeof safety['message'] === 'string' ? safety['message'] : '',
    comparison_id: typeof safety['comparison_id'] === 'string' ? safety['comparison_id'] : '',
    comparison_status:
      typeof safety['comparison_status'] === 'string' ? safety['comparison_status'] : '',
    gate_reasons: readStringArray(safety['gate_reasons']),
    required_action: typeof safety['required_action'] === 'string' ? safety['required_action'] : '',
  }
}

export function readPlanDraftState(  payload: Record<string, unknown>,
): V2PlanDraftState | null {
  const stage = payload['stage']
  if (typeof stage !== 'string' || !V2_PLAN_DRAFT_STAGES.has(stage as V2PlanDraftStage)) {
    return null
  }
  const rawDraft = payload['draft']
  const draft = (rawDraft && typeof rawDraft === 'object' ? rawDraft : {}) as Record<string, unknown>
  const participants = draft['participants']
  const routeSelection = readRouteSelection(payload['route_selection'])
  return {
    stage: stage as V2PlanDraftStage,
    draft_id: typeof payload['draft_id'] === 'string' ? payload['draft_id'] : '',
    conversation_id: typeof payload['conversation_id'] === 'string' ? payload['conversation_id'] : '',
    source: payload['source'] === 'natural_language' ? 'natural_language' : 'structured_intent',
    intent: typeof payload['intent'] === 'string' ? payload['intent'] : '',
    // Ticket 02 AC3: the rendered draft summary. Absent on an older payload,
    // which is an empty summary — never a guess and never a dropped event.
    summary: typeof payload['summary'] === 'string' ? payload['summary'] : '',
    // Ticket 04 AC5: the draft type. A payload that names one must name a known
    // value; a payload from a server that predates the field is answered by the
    // same rule the server uses — a draft pointing at nothing verified is a
    // generic draft — so the reader never guesses a route plan into existence.
    plan_type: readPlanType(payload['plan_type'], routeSelection),
    // Ticket 03: the route reference. Both ids must be non-empty strings or the
    // draft has no reference — a half-read pointer would render a route the
    // server never issued, which is the AC4 failure this reader exists to stop.
    route_selection: routeSelection,
    // Ticket 05 AC1: the server's missing-field enumeration. This is the SAME
    // reader for the SSE payload and for the durable localStorage mirror, so a
    // checklist that survived a refresh is still the checklist the summary
    // spells out — and a malformed entry is dropped rather than guessed into a
    // 待确认 / 待核验 state the server never claimed.
    missing_fields: readMissingPlanFields(payload['missing_fields']),
    // Ticket 06 AC5: the verdict, read by the same strict reader for the SSE
    // payload and the durable mirror. Unknown verdict → null (say nothing),
    // never a guessed one.
    rules: readPlanRules(payload['rules']),
    // Ticket 07 AC5: the gate's refusal, read by the same strict reader for the
    // SSE payload and the durable mirror, so a safety block explained in the
    // thread is still explained after a refresh — and an unknown code never
    // becomes an invented reason.
    safety: readPlanSafety(payload['safety']),
    draft: {
      destination_or_route:
        typeof draft['destination_or_route'] === 'string' ? draft['destination_or_route'] : '',
      activity_date: typeof draft['activity_date'] === 'string' ? draft['activity_date'] : '',
      participants: typeof participants === 'number' && Number.isFinite(participants)
        ? participants
        : null,
      // Ticket 06 AC1/AC5: the collected supplementary arrangements. Absent on
      // a server that predates them, malformed in a hand-edited mirror, or
      // simply not filled in — all read as the all-empty record, never as a
      // partially guessed one.
      arrangements: readPlanArrangements(draft['arrangements']),
    },
  }
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
