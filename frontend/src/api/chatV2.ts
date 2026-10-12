/**
 * Chat v2 client — the only chat链路 (v1 removed).
 *
 * Ticket 08: spec AC1 requires "Frontend 使用独立 v2 client 和 version 2
 * event reducer，不把原始框架事件或 v1 event union 当作 v2 契约". This
 * module owns the v2 wire types, SSE parser, and stream connection.
 *
 * Parser guarantees (spec AC6):
 *   - CRLF frame separators
 *   - fragmented frames (chunks split mid-frame)
 *   - multiple frames per chunk
 *   - invalid JSON → dropped (counted, not thrown)
 *   - unknown event types → dropped (counted)
 *   - duplicate event id → ignored after first
 *   - exactly one terminal event ends the stream
 */

import { API, API_ORIGIN } from './config'

// ---- Wire types ---------------------------------------------------------

// Spec decision #27: public events are limited to run start, user-safe
// progress, assistant delta, tool start/completion, approval required/
// resolved, artifact available, memory consent required, pause, terminal
// outcomes, and heartbeat. Ticket 11 adds approval.required /
// approval.resolved / run.paused so the client can drive the durable
// approval flow end-to-end with the server.
export type V2EventType =
  | 'run_started'
  | 'progress'
  | 'text'
  | 'tool_call'
  | 'tool_result'
  | 'artifact_candidate'
  | 'artifact.available'
  | 'approval.required'
  | 'approval.resolved'
  | 'memory.consent_required'
  | 'plan.draft_state'
  | 'run.paused'
  | 'effect.outcome_unknown'
  | 'error'
  | 'done'
  | 'cancelled'
  | 'heartbeat'

// ---- Plan book (Ticket 01) ------------------------------------------------

/**
 * The whitelisted structured intents the server accepts on
 * ``POST /api/v2/chat/sse``. Quick actions send these directly; free text is
 * only ever normalised by server-side rules. An intent outside this list is
 * rejected by the server's request model before streaming starts, so the
 * frontend must never invent one.
 */
export type V2PlanIntent =
  | 'plan_create'
  | 'plan_confirm'
  | 'plan_cancel'
  | 'plan_regenerate'

/**
 * Draft stages of ``plan.draft_state`` — the same vocabulary as
 * ``PlanDraftStage`` on the server and ``plan_drafts.stage`` in SQLite.
 *
 * ``awaiting_confirmation`` is the ambiguous-intent parking lot: the server
 * asked whether to generate a plan book, and the dialog MUST stay closed
 * until the user confirms. Only ``awaiting_input`` opens it.
 */
export type V2PlanDraftStage =
  | 'awaiting_confirmation'
  | 'awaiting_input'
  | 'minimal_draft'
  | 'cancelled'

/** The three fields the entry dialog collects. Absent values stay absent. */
export interface V2PlanDraftFields {
  destination_or_route: string
  activity_date: string
  participants: number | null
}

export interface V2PlanDraftState {
  stage: V2PlanDraftStage
  draft_id: string
  conversation_id: string
  /** Whether a structured intent or the deterministic rules opened the flow. */
  source: 'structured_intent' | 'natural_language'
  intent: string
  /**
   * Ticket 02 AC3: the server-rendered draft summary for this stage and these
   * fields (``plan_workflow.draft_summary``). It travels inside the draft
   * state, so the dialog's read-only block and the chat bubble show the same
   * text, and a restored draft still shows its summary after a refresh.
   * Empty only for a payload from a server that predates this field.
   */
  summary: string
  draft: V2PlanDraftFields
}

export const V2_PLAN_DRAFT_STAGES: ReadonlySet<V2PlanDraftStage> = new Set([
  'awaiting_confirmation',
  'awaiting_input',
  'minimal_draft',
  'cancelled',
])

/**
 * artifact.available payload (A02): the only public artifact shape. No
 * sandbox path, no host storage path — download goes through
 * /api/v2/artifacts/{artifact_id}.
 */
export interface V2Artifact {
  artifact_id: string
  public_filename: string
  mime_type: string
  size_bytes: number
  sha256?: string
  expires_at?: string
  message?: string
}

// Run-terminal events: the run is in a final state and no further
// continuation is possible. The reducer's sticky guard refuses to
// transition out of these states (defending against a buggy server).
export const TERMINAL_V2_EVENT_TYPES: ReadonlySet<V2EventType> = new Set([
  'error',
  'done',
  'cancelled',
])

// Stream-terminal events: the SSE response body ends after one of these
// is emitted. ``run.paused`` is stream-terminal (the server closes the
// connection so the client must POST /approvals/{id}/decision then open
// /chat/resume/{run_id}) but NOT run-terminal — the run is still alive
// and awaiting user input. The stream loop uses this set to decide when
// to stop reading; the reducer uses TERMINAL_V2_EVENT_TYPES for the
// sticky guard so ``paused`` → ``running`` remains legal on resume.
export const STREAM_TERMINAL_V2_EVENT_TYPES: ReadonlySet<V2EventType> = new Set([
  'error',
  'done',
  'cancelled',
  'run.paused',
])

export interface V2EventEnvelope {
  version: '2'
  sequence: number
  type: V2EventType
  payload: Record<string, unknown>
  request_id?: string
  conversation_id?: string
  /**
   * Server-generated run id (spec decision #26). Empty when v2 persistence
   * is off; otherwise the frontend uses it for /snapshot, /cancel, /events.
   */
  run_id?: string
  /**
   * Server-generated event UUID (spec decision #26). Unique per event.
   * Used for deduplication and audit tracing.
   */
  event_id?: string
  /**
   * UTC ISO 8601 timestamp (spec decision #26). Set by EventStream.build.
   */
  timestamp?: string
}

export interface V2StreamHandlers {
  onEvent: (event: V2EventEnvelope) => void
  onError: (error: string) => void
  onOpen?: () => void
  onTerminal?: (event: V2EventEnvelope) => void
}

export interface V2StreamOptions {
  body: string
  onEvent: (event: V2EventEnvelope) => void
  onError: (error: string) => void
  onOpen?: () => void
  onTerminal?: (event: V2EventEnvelope) => void
}

// ---- Parser -------------------------------------------------------------

/**
 * Parse a single SSE frame (one or more `data:` lines separated by `\r?\n`)
 * into a v2 envelope or null.
 *
 * Returns null for:
 *   - empty frames (keepalive comments, blank lines)
 *   - invalid JSON
 *   - valid JSON that is not a v2 envelope (missing version/sequence/type)
 *   - envelopes with unknown event types (dropped + counted via the
 *     `onDropped` callback if provided)
 *
 * Spec AC6: parser covers CRLF, fragmented frame, multi frame, invalid JSON,
 * unknown event, duplicate event, and唯一 terminal event. The parser itself
 * is pure; duplicate/terminal tracking lives in the stream loop where we
 * have state across frames.
 */
export function parseV2Frame(
  frame: string,
  onDropped?: (reason: 'invalid_json' | 'unknown_type' | 'not_envelope', raw: string) => void,
): V2EventEnvelope | null {
  // Aggregate all `data:` lines per SSE spec. CRLF and LF both supported.
  const dataLines: string[] = []
  for (const rawLine of frame.split(/\r\n|\n/)) {
    const line = rawLine.replace(/\r$/, '')
    if (line.startsWith('data:')) {
      dataLines.push(line.slice(5).replace(/^ /, ''))
    } else if (line.startsWith(':')) {
      // SSE comment / keepalive — ignore.
      continue
    }
    // Other field names (event:, id:, retry:) are not used by v2.
  }

  if (dataLines.length === 0) return null
  const jsonStr = dataLines.join('\n').trim()
  if (!jsonStr) return null

  let parsed: unknown
  try {
    parsed = JSON.parse(jsonStr)
  } catch {
    onDropped?.('invalid_json', jsonStr)
    return null
  }

  if (typeof parsed !== 'object' || parsed === null) {
    onDropped?.('not_envelope', jsonStr)
    return null
  }
  const obj = parsed as Record<string, unknown>
  if (obj.version !== '2' || typeof obj.sequence !== 'number' || typeof obj.type !== 'string') {
    onDropped?.('not_envelope', jsonStr)
    return null
  }
  // Unknown event types are dropped but counted so monitoring can detect
  // adapter drift. The v2 server is the only producer; an unknown type
  // means the adapter is emitting events outside the public contract.
  if (!isV2EventType(obj.type)) {
    onDropped?.('unknown_type', jsonStr)
    return null
  }
  if (typeof obj.payload !== 'object' || obj.payload === null) {
    // Defensive: payload is required by the v2 contract. Treat as invalid.
    onDropped?.('not_envelope', jsonStr)
    return null
  }
  return {
    version: '2',
    sequence: obj.sequence,
    type: obj.type,
    payload: obj.payload as Record<string, unknown>,
    request_id: typeof obj.request_id === 'string' ? obj.request_id : undefined,
    conversation_id: typeof obj.conversation_id === 'string' ? obj.conversation_id : undefined,
    run_id: typeof obj.run_id === 'string' ? obj.run_id : undefined,
    event_id: typeof obj.event_id === 'string' ? obj.event_id : undefined,
    timestamp: typeof obj.timestamp === 'string' ? obj.timestamp : undefined,
  }
}

function isV2EventType(value: unknown): value is V2EventType {
  return (
    value === 'run_started' ||
    value === 'progress' ||
    value === 'text' ||
    value === 'tool_call' ||
    value === 'tool_result' ||
    value === 'artifact_candidate' ||
    value === 'artifact.available' ||
    value === 'approval.required' ||
    value === 'approval.resolved' ||
    value === 'memory.consent_required' ||
    value === 'plan.draft_state' ||
    value === 'run.paused' ||
    value === 'effect.outcome_unknown' ||
    value === 'error' ||
    value === 'done' ||
    value === 'cancelled' ||
    value === 'heartbeat'
  )
}

// ---- Stream connection --------------------------------------------------

/**
 * Open a v2 SSE stream. Returns a cleanup function.
 *
 * Duplicate event detection: spec AC6 requires duplicate event id handling.
 * The stream tracks seen sequences and ignores repeats after the first.
 * Terminal handling: exactly one terminal event ends the stream; subsequent
 * frames (if any) are ignored.
 */
export function createV2StreamConnection(
  url: string,
  options: V2StreamOptions,
): () => void {
  let aborted = false
  let reader: ReadableStreamDefaultReader<Uint8Array> | null = null
  const controller = new AbortController()
  const seenSequences = new Set<number>()
  let terminalSeen = false

  const cleanup = () => {
    aborted = true
    controller.abort()
    reader?.cancel().catch(() => {
      // Stream may already be closed.
    })
  }

  const dispatch = (event: V2EventEnvelope): boolean => {
    // Spec AC6: duplicate event id → ignore after first.
    if (seenSequences.has(event.sequence)) {
      return false
    }
    seenSequences.add(event.sequence)

    options.onEvent(event)

    // Ticket 11: the stream terminates on ``run.paused`` too — the
    // server closes the SSE response after emitting it so the client
    // must POST /approvals/{id}/decision and re-open /chat/resume.
    // ``onTerminal`` fires for every stream-terminal event; callers
    // distinguish run-terminal (error/done/cancelled) from
    // stream-only-terminal (run.paused) by inspecting ``event.type``.
    if (STREAM_TERMINAL_V2_EVENT_TYPES.has(event.type)) {
      // Spec AC6: exactly one terminal event ends the stream.
      if (terminalSeen) {
        // Programmer error on the server side; ignore the second terminal.
        return true
      }
      terminalSeen = true
      options.onTerminal?.(event)
      return true
    }
    return false
  }

  const run = async () => {
    try {
      const response = await fetch(url, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
        },
        body: options.body,
        signal: controller.signal,
      })

      if (!response.ok) {
        const message = await v2ResponseErrorMessage(response)
        if (!aborted) options.onError(message)
        return
      }

      options.onOpen?.()

      reader = response.body?.getReader() ?? null
      if (!reader) {
        if (!aborted) options.onError('Response body is not readable')
        return
      }

      const decoder = new TextDecoder()
      let buffer = ''

      while (!aborted) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        // Spec AC6: split on blank line (LF or CRLF). Keep the trailing
        // fragment in buffer to handle chunks split mid-frame.
        const frames = buffer.split(/\r?\n\r?\n/)
        buffer = frames.pop() || ''

        for (const frame of frames) {
          if (!frame.trim()) continue
          const event = parseV2Frame(frame)
          if (event && dispatch(event) && terminalSeen) {
            // Terminal reached: stop reading.
            cleanup()
            return
          }
        }
      }

      // Flush any trailing frame (e.g. terminal without final blank line).
      const remaining = `${buffer}${decoder.decode()}`
      if (remaining.trim() && !aborted && !terminalSeen) {
        const event = parseV2Frame(remaining)
        if (event) dispatch(event)
      }

      // Spec AC: transport closure is NOT a terminal. If we reach here
      // without a terminal, the caller must treat the run as
      // outcome_unknown and reconcile via /snapshot + /events.
      if (!aborted && !terminalSeen) {
        options.onError('stream_closed_without_terminal')
      }
    } catch (error) {
      if (!aborted && !(error instanceof DOMException && error.name === 'AbortError')) {
        options.onError(error instanceof Error ? error.message : 'Connection failed')
      }
    }
  }

  void run()

  return cleanup
}

async function v2ResponseErrorMessage(response: Response): Promise<string> {
  const fallback = `HTTP ${response.status}: ${response.statusText || '请求失败'}`
  const text = await response.text().catch(() => '')
  if (!text.trim()) return fallback
  try {
    const payload = JSON.parse(text) as Record<string, unknown>
    const message = payload.detail ?? payload.error ?? payload.message
    if (typeof message === 'string' && message.trim()) {
      return `HTTP ${response.status}: ${message.trim()}`
    }
  } catch {
    // Plain-text error bodies are useful as-is.
  }
  return `HTTP ${response.status}: ${text.trim()}`
}

// ---- Run snapshot / events / cancel -------------------------------------

export interface V2RunSnapshot {
  run_id: string
  request_id: string
  conversation_id: string
  profile_id: string
  // Ticket 11: ``paused`` is a non-terminal run state — the run is
  // awaiting user approval and will resume on
  // POST /api/v2/approvals/{approval_id}/decision.
  status: 'running' | 'paused' | 'done' | 'error' | 'cancelled'
  terminal_type: string | null
  created_at: string
  updated_at: string
  terminal_at: string | null
  sequence_cursor: number
  messages: Array<{ role: 'user' | 'assistant'; content: string }>
  pending_actions: unknown[]
}

export interface V2ReplayResponse {
  run_id: string
  from_seq: number
  events: Array<{
    version: '2'
    sequence: number
    type: V2EventType
    payload: Record<string, unknown>
    request_id: string
    conversation_id: string
    run_id: string
    event_id: string
    timestamp: string
  }>
}

// ---- Recovery planning (A06) ----------------------------------------------

/** Run-level terminal statuses, as returned by /snapshot and persisted as v2_status. */
export const V2_TERMINAL_RUN_STATUSES: ReadonlySet<string> = new Set([
  'done',
  'error',
  'cancelled',
])

export type V2RecoveryPlan =
  | { kind: 'skip' } // no usable id — nothing to recover, keep local state
  | { kind: 'local-terminal' } // persisted status already terminal — trust it, skip the network
  | { kind: 'clear' } // server has no record — drop the local recovery context
  | { kind: 'replay-terminal' } // server terminal — replay once and finalize
  | { kind: 'replay-paused' } // server paused — replay and re-show the approval prompt
  | { kind: 'poll' } // server still running — replay, then poll the snapshot

/**
 * Pure recovery decision (A06). Fed the persisted run context and the
 * snapshot status (``null`` when the snapshot 404'd or was not fetched).
 * The recovery effect in SuperAgent executes the returned plan.
 *
 * A running run cannot be re-attached: a duplicate POST /chat/sse with
 * the same request_id replays stored events and closes the stream
 * WITHOUT a terminal (the run is not done server-side), which the
 * client would report as ``stream_closed_without_terminal``. Polling
 * the snapshot until the run reaches a terminal or pauses is the
 * recovery path instead.
 */
export function planV2Recovery(
  persisted: { run_id?: string; request_id?: string; v2_status?: string },
  snapshotStatus: string | null,
): V2RecoveryPlan {
  // Empty run_id (refresh before the first event) with no request_id
  // either — nothing identifies the run; keep the local context so the
  // server-side run is not orphaned by a local reset.
  if (!persisted.run_id && !persisted.request_id) return { kind: 'skip' }
  // Ticket 08 documented contract: a locally terminal run skips the
  // snapshot fetch entirely (avoids the 404 race when the server reaped
  // the run row).
  if (persisted.v2_status && V2_TERMINAL_RUN_STATUSES.has(persisted.v2_status)) {
    return { kind: 'local-terminal' }
  }
  if (snapshotStatus === null) return { kind: 'clear' }
  if (V2_TERMINAL_RUN_STATUSES.has(snapshotStatus)) return { kind: 'replay-terminal' }
  if (snapshotStatus === 'paused') return { kind: 'replay-paused' }
  return { kind: 'poll' }
}

export interface V2CancelResponse {
  accepted: boolean
  reason: string
  run_id: string
}

export interface V2MemorySettings {
  subject_id: string
  enabled: boolean
}

export interface V2MemoryCandidate {
  candidate_id: string
  category: string
  sensitivity: string
  preview: string
  expires_at: string
}

export interface V2Memory {
  memory_id: string
  category: string
  content: string
  created_at: string
  updated_at: string
  expires_at: string | null
  consent: {
    type: string
    consented_at: string
    candidate_id: string | null
  }
}

export interface V2MemoryPage {
  items: V2Memory[]
  total: number
  limit: number
  offset: number
}

export async function submitV2MemoryDecision(
  candidateId: string,
  decision: 'approve' | 'reject',
  decisionRequestId: string,
): Promise<Record<string, unknown>> {
  const resp = await fetch(API.chatV2MemoryCandidateDecision(candidateId), {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify({ decision, decision_request_id: decisionRequestId }),
  })
  if (!resp.ok) throw new Error(await v2ResponseErrorMessage(resp))
  return (await resp.json()) as Record<string, unknown>
}

export async function fetchV2Memories(limit = 100, offset = 0): Promise<V2MemoryPage> {
  const resp = await fetch(API.chatV2Memories(limit, offset), {
    method: 'GET',
    credentials: 'include',
    headers: { Accept: 'application/json' },
  })
  if (!resp.ok) throw new Error(await v2ResponseErrorMessage(resp))
  return (await resp.json()) as V2MemoryPage
}

export async function updateV2Memory(
  memoryId: string,
  update: { content: string; category: string; ttl_days: number | null },
): Promise<V2Memory> {
  const resp = await fetch(API.chatV2Memory(memoryId), {
    method: 'PATCH',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(update),
  })
  if (!resp.ok) throw new Error(await v2ResponseErrorMessage(resp))
  return (await resp.json()) as V2Memory
}

export async function deleteV2Memory(memoryId: string): Promise<void> {
  const resp = await fetch(API.chatV2Memory(memoryId), {
    method: 'DELETE',
    credentials: 'include',
    headers: { Accept: 'application/json' },
  })
  if (!resp.ok) throw new Error(await v2ResponseErrorMessage(resp))
}

export async function clearV2Memories(): Promise<number> {
  const resp = await fetch(`${API.chatV2MemoryClear}?confirm=true`, {
    method: 'DELETE',
    credentials: 'include',
    headers: { Accept: 'application/json' },
  })
  if (!resp.ok) throw new Error(await v2ResponseErrorMessage(resp))
  const result = (await resp.json()) as { deleted: number }
  return result.deleted
}

export async function exportV2Memories(): Promise<{ artifact_id: string; expires_at: string }> {
  const resp = await fetch(API.chatV2MemoryExport, {
    method: 'GET',
    credentials: 'include',
    headers: { Accept: 'application/json' },
  })
  if (!resp.ok) throw new Error(await v2ResponseErrorMessage(resp))
  return (await resp.json()) as { artifact_id: string; expires_at: string }
}

export async function fetchV2MemorySettings(): Promise<V2MemorySettings> {
  const resp = await fetch(API.chatV2MemorySettings, {
    method: 'GET',
    credentials: 'include',
    headers: { Accept: 'application/json' },
  })
  if (!resp.ok) throw new Error(`HTTP ${resp.status}: 记忆设置获取失败`)
  return (await resp.json()) as V2MemorySettings
}

export async function updateV2MemorySettings(enabled: boolean): Promise<V2MemorySettings> {
  const resp = await fetch(API.chatV2MemorySettings, {
    method: 'PUT',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify({ enabled }),
  })
  if (!resp.ok) throw new Error(`HTTP ${resp.status}: 记忆设置更新失败`)
  return (await resp.json()) as V2MemorySettings
}

/** Fetch the run snapshot for refresh recovery. Returns null on 404. */
export async function fetchV2Snapshot(runId: string): Promise<V2RunSnapshot | null> {
  const resp = await fetch(API.chatV2Snapshot(runId), {
    method: 'GET',
    credentials: 'include',
    headers: { Accept: 'application/json' },
  })
  if (resp.status === 404) return null
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status}: ${resp.statusText || '快照获取失败'}`)
  }
  return (await resp.json()) as V2RunSnapshot
}

/** Replay stored events for refresh recovery. */
export async function fetchV2Events(
  runId: string,
  fromSeq: number,
): Promise<V2ReplayResponse> {
  const resp = await fetch(API.chatV2Events(runId, fromSeq), {
    method: 'GET',
    credentials: 'include',
    headers: { Accept: 'application/json' },
  })
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status}: ${resp.statusText || '事件回放失败'}`)
  }
  return (await resp.json()) as V2ReplayResponse
}

/** Idempotent cancel. Returns the outcome (accepted or already_terminal). */
export async function cancelV2Run(
  runId: string,
  idempotencyKey: string,
): Promise<V2CancelResponse> {
  const resp = await fetch(API.chatV2Cancel(runId), {
    method: 'POST',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify({ idempotency_key: idempotencyKey }),
  })
  // 200 = accepted, 409 = already terminal — both are valid outcomes.
  if (resp.status !== 200 && resp.status !== 409) {
    throw new Error(`HTTP ${resp.status}: ${resp.statusText || '取消失败'}`)
  }
  return (await resp.json()) as V2CancelResponse
}

// ---- Ticket 11: approvals + resume -------------------------------------

/**
 * Approval payload carried by ``approval.required`` events and surfaced
 * to the UI for the user to approve or reject. Mirrors the server-side
 * ``ApprovalRequest`` shape defined in ``agent_v2/approvals.py``.
 */
export interface V2ApprovalRequest {
  approval_id: string
  tool: string
  args: Record<string, unknown>
  risk_level: 'low' | 'medium' | 'high'
  impact: string
  expires_at: string
  message?: string
}

/**
 * Response body for POST /api/v2/approvals/{approval_id}/decision.
 *
 * The server returns 200 on success (including idempotent duplicates of
 * the same decision_request_id). 409 covers already-resolved, expired,
 * run-terminal, run-not-paused, and sandbox-lease-lost cases — the
 * caller inspects ``reason`` to render the right UI.
 */
export interface V2ApprovalDecisionResponse {
  approval_id: string
  run_id: string
  tool: string
  decision: 'approve' | 'reject'
  status: 'approved' | 'rejected'
  idempotent: boolean
  /** Path the client should open as an SSE stream to continue the run. */
  resume_url: string
}

/**
 * Submit an approve/reject decision for a pending approval.
 *
 * Spec AC4 + AC5: the decision is idempotent w.r.t.
 * ``decisionRequestId`` — a duplicate POST with the same key returns the
 * original outcome with ``idempotent=true``. The caller then opens
 * ``resume_url`` as an SSE stream to receive the continuation
 * (``approval.resolved`` → tool execution → terminal).
 *
 * Status codes:
 *   - 200 — decision recorded (or idempotent duplicate)
 *   - 404 — approval_id unknown
 *   - 409 — already resolved / expired / run terminal / run not paused /
 *     sandbox lease lost (``reason`` field explains which)
 *   - 422 — invalid body
 *
 * On 409 the caller surfaces a user-facing message based on ``reason``;
 * on other non-200 statuses the function throws so the caller can
 * display a generic error.
 */
export async function submitV2Approval(
  approvalId: string,
  decision: 'approve' | 'reject',
  decisionRequestId: string,
): Promise<V2ApprovalDecisionResponse> {
  const resp = await fetch(API.chatV2ApprovalDecision(approvalId), {
    method: 'POST',
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify({
      decision,
      decision_request_id: decisionRequestId,
    }),
  })
  if (resp.status === 200) {
    return (await resp.json()) as V2ApprovalDecisionResponse
  }
  if (resp.status === 409) {
    // 409 carries a structured reason so the caller can render the
    // right UI (e.g. "expired" vs "already_resolved"). We rethrow with
    // the reason prefix so the caller can match on it.
    const body = (await resp.json().catch(() => ({}))) as { reason?: string; message?: string }
    const reason = body.reason || 'unknown'
    const message = body.message || '审批决策失败'
    throw new Error(`409:${reason}:${message}`)
  }
  if (resp.status === 404) {
    throw new Error('404:approval_not_found:审批不存在或已被清理。')
  }
  throw new Error(`HTTP ${resp.status}: ${resp.statusText || '审批决策失败'}`)
}

/**
 * Build the SSE URL for POST /api/v2/chat/resume/{run_id} (A10).
 *
 * Prefers the server-issued ``resume_url`` from the decision response —
 * the backend owns the gateway path, so the client should not re-derive
 * it — and appends the replay cursor. Falls back to the documented
 * resume path when the field is absent.
 *
 * The resume endpoint is itself an SSE stream: the server replays
 * stored events with sequence > ``fromSeq``, then emits the
 * continuation (``approval.resolved`` → tool execution → terminal).
 * The caller feeds this URL to ``createV2StreamConnection`` exactly
 * like the initial /chat/sse URL — the only difference is the method
 * path and the ``from_seq`` query parameter.
 */
export function buildV2ResumeUrlFromDecision(
  decision: Pick<V2ApprovalDecisionResponse, 'resume_url' | 'run_id'>,
  fromSeq: number,
): string {
  // Same-origin path via the shared API map; strip the builder's query
  // so the cursor below stays the only parameter.
  const fallback = API.chatV2Resume(decision.run_id, 0).split('?')[0]
  let base =
    typeof decision.resume_url === 'string' && decision.resume_url
      ? decision.resume_url
      : fallback
  // A10 review fix: the server issues a root-relative path; the stream
  // connection fetches it bare, so a configured API origin must be
  // prefixed here or the resume request would hit the frontend origin.
  if (base.startsWith('/') && API_ORIGIN) {
    base = `${API_ORIGIN}${base}`
  }
  const separator = base.includes('?') ? '&' : '?'
  return `${base}${separator}from_seq=${fromSeq}`
}
