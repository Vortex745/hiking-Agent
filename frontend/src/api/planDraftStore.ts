/**
 * Ticket 02 AC2/AC4 — the conversation-keyed durable draft store.
 *
 * Why this exists instead of reusing the run record: ``V2PersistedRun`` is
 * nulled at every run terminal (``handleV2Terminal``), so the draft mirrored
 * onto it only survives a refresh *mid-run*. A plan draft outlives its run by
 * design (server-side it lives on the conversation axis, in ``plan_drafts``),
 * so the client's restore source has to be keyed the same way — by
 * ``conversation_id`` — and stay untouched when a run terminates.
 *
 * Three rules this module enforces, because each one is a silent failure
 * otherwise:
 *
 * 1. Entries are re-read through ``readPlanDraftState``, the SAME strict
 *    reader the SSE payload goes through. A stage this build does not know
 *    (a draft written by an older/newer frontend) restores as ``null`` rather
 *    than opening a dialog on a guess.
 * 2. ``cancelled`` is not restorable — reading or writing it clears the slot,
 *    which is AC4's "取消清理未完成的计划状态" on the client.
 * 3. An entry whose stored ``conversation_id`` disagrees with the key is
 *    dropped, so a draft can never leak into another conversation.
 */

import type { V2PlanDraftStage, V2PlanDraftState } from './chatV2'
import { readPlanDraftState } from './v2Reducer'

const PLAN_DRAFTS_KEY = 'ai-hiking-agent-plan-drafts'

/** Same cap the session list uses, so this map cannot grow without bound. */
const MAX_STORED_DRAFTS = 50

/**
 * Stages the user can reopen the填写 dialog from — the stages whose draft has
 * an unfinished form. ``awaiting_confirmation`` is deliberately absent: that
 * stage is answered by the confirm prompt, and opening a form from it would
 * bypass spec AC2. ``cancelled`` is terminal.
 */
export const OPEN_PLAN_DRAFT_STAGES: ReadonlySet<V2PlanDraftStage> = new Set([
  'awaiting_input',
  'minimal_draft',
])

/** A cancelled draft is terminal: nothing durable, nothing restorable. */
export function isDurablePlanDraftStage(stage: V2PlanDraftStage): boolean {
  return stage !== 'cancelled'
}

/** Whether the dialog may be reopened from this stage without a server round-trip. */
export function isReopenablePlanDraftStage(stage: V2PlanDraftStage): boolean {
  return OPEN_PLAN_DRAFT_STAGES.has(stage)
}

type DraftMap = Record<string, V2PlanDraftState>

function readMap(): DraftMap {
  try {
    const raw = localStorage.getItem(PLAN_DRAFTS_KEY)
    if (!raw) return {}
    const parsed = JSON.parse(raw) as unknown
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return {}
    const entries = Object.entries(parsed as Record<string, unknown>)
      .filter(([, value]) => Boolean(value) && typeof value === 'object')
    return Object.fromEntries(entries) as DraftMap
  } catch {
    return {}
  }
}

function writeMap(map: DraftMap): void {
  try {
    // Re-insertion order is the recency order the cap below keeps.
    const trimmed = Object.fromEntries(Object.entries(map).slice(-MAX_STORED_DRAFTS))
    localStorage.setItem(PLAN_DRAFTS_KEY, JSON.stringify(trimmed))
  } catch { /* ignore — a full or blocked store must not break the thread */ }
}

/**
 * The conversation's stored draft, or ``null`` when there is nothing to
 * restore. Never throws and never guesses: a corrupt entry, an unknown stage,
 * a cancelled draft or a mismatched conversation all clear the slot.
 */
export function loadPlanDraft(conversationId: string): V2PlanDraftState | null {
  if (!conversationId) return null
  const entry = readMap()[conversationId]
  if (!entry) return null

  // A stored entry is untrusted input (it may come from another build of this
  // app), so it goes through the same strict reader as the SSE payload —
  // including its stage check.
  const state = readPlanDraftState(entry as unknown as Record<string, unknown>)
  if (!state || !isDurablePlanDraftStage(state.stage)) {
    clearPlanDraft(conversationId)
    return null
  }
  if (state.conversation_id && state.conversation_id !== conversationId) {
    clearPlanDraft(conversationId)
    return null
  }
  // The key wins: a draft written before the conversation rotated must not
  // come back carrying the old id.
  return { ...state, conversation_id: conversationId }
}

/** Mirror the current draft. A cancelled stage clears instead of storing. */
export function savePlanDraft(state: V2PlanDraftState): void {
  const conversationId = state.conversation_id
  if (!conversationId) return
  if (!isDurablePlanDraftStage(state.stage)) {
    clearPlanDraft(conversationId)
    return
  }
  const map = readMap()
  delete map[conversationId]
  map[conversationId] = state
  writeMap(map)
}

/** Drop the conversation's stored draft (AC4). Idempotent. */
export function clearPlanDraft(conversationId: string): void {
  if (!conversationId) return
  const map = readMap()
  if (!(conversationId in map)) return
  delete map[conversationId]
  writeMap(map)
}

/**
 * Field equality, used to keep ``setPlanDraft`` from churning object identity
 * on every restore. Pure, so it is safe inside a state updater.
 */
export function isSamePlanDraft(
  a: V2PlanDraftState | null,
  b: V2PlanDraftState | null,
): boolean {
  if (a === b) return true
  if (!a || !b) return false
  return a.stage === b.stage
    && a.draft_id === b.draft_id
    && a.conversation_id === b.conversation_id
    && a.summary === b.summary
    && a.draft.destination_or_route === b.draft.destination_or_route
    && a.draft.activity_date === b.draft.activity_date
    && a.draft.participants === b.draft.participants
}
