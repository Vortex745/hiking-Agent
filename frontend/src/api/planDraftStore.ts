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
 *
 * Ticket 04's ``awaiting_generic_choice`` is absent for the same reason
 * ``awaiting_confirmation`` is: it is answered by its own choice panel, so
 * making it reopenable would stack two absolutely-positioned ``top-16``
 * panels on top of each other.
 *
 * Ticket 05's two review stages ARE in the set, deliberately: the draft they
 * carry has unfinished arrangements by definition (that is the missing-field
 * list), the user must be able to reopen the form to fill them, and 取消草稿
 * has to reach the server from either stage. Unlike the parked research stage
 * they render no second panel — the review affordances live inside this one
 * panel, so nothing stacks.
 */
export const OPEN_PLAN_DRAFT_STAGES: ReadonlySet<V2PlanDraftStage> = new Set([
  'awaiting_input',
  'minimal_draft',
  'awaiting_first_review',
  'first_reviewed',
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
    // Ticket 04 AC5: the draft's type is compared like every other field. Skip
    // it and a route_plan → generic_draft flip (or the reverse) looks like "no
    // change", so the restore effect hands back the OLD object and the marker
    // the user is supposed to see silently disappears after a refresh.
    && a.plan_type === b.plan_type
    && isSameRouteSelection(a.route_selection, b.route_selection)
    // Ticket 05 AC1: the missing-field list is compared like every other field.
    // Skip it and a draft whose gaps just changed (one arrangement filled in,
    // a route basis re-read) looks like "no change" to the restore effect, so
    // the checklist the user is deciding from silently keeps the OLD entries
    // after a refresh.
    && isSameMissingFields(a.missing_fields, b.missing_fields)
    && a.draft.destination_or_route === b.draft.destination_or_route
    && a.draft.activity_date === b.draft.activity_date
    && a.draft.participants === b.draft.participants
}

/**
 * Ticket 03: the route reference is compared field by field, like every other
 * draft field. Skipping it here would make a selection look like "no change",
 * so the restore effect would hand back the OLD object and the reference — and
 * with it the route basis — would silently disappear after a refresh, which
 * reads like a store bug rather than a wiring bug.
 */
function isSameRouteSelection(
  a: V2PlanDraftState['route_selection'],
  b: V2PlanDraftState['route_selection'],
): boolean {
  if (a === b) return true
  if (!a || !b) return false
  return a.comparison_id === b.comparison_id && a.candidate_id === b.candidate_id
}

/**
 * Ticket 05 AC1: the missing-field list compared entry by entry, in order. The
 * key is what identifies an entry (the label and status come from the same
 * server render), so equality is key + status: a field that moved from 待确认
 * to filled, or a fact the gate stopped certifying, is a change the user has to
 * see.
 */
function isSameMissingFields(
  a: V2PlanDraftState['missing_fields'],
  b: V2PlanDraftState['missing_fields'],
): boolean {
  if (a === b) return true
  if (a.length !== b.length) return false
  return a.every((entry, index) => {
    const other = b[index]
    return entry.key === other.key && entry.status === other.status
  })
}
