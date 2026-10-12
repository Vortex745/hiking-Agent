import { useEffect, useState } from 'react'
import Button from './Button'
import X from 'reicon-react/icons/X'
import type { V2PlanDraftFields } from '../api/chatV2'

/**
 * Ticket 01 — 计划书入口弹窗.
 *
 * There is no reusable Dialog in this project (the only real modal is the
 * image preview in ``GeminiThread.tsx``), so this component imitates that
 * modal's conventions instead of adding a portal dependency: ``fixed
 * inset-0 z-[100]`` over a blurred scrim, ``role="dialog"`` +
 * ``aria-modal="true"``, and an Escape listener bound while open.
 *
 * It collects exactly the three minimal draft fields the spec names —
 * 目的地或路线 / 活动日期 / 参与人数 — and nothing else. The checks below
 * mirror the server's (:class:`agent_v2.plan_workflow.PlanDraftInput`) so
 * a typo is caught locally, while the server stays the authority: a draft
 * that bypasses them is still rejected with a stable reason. Values the
 * user leaves empty stay empty — the form never prefills an invented
 * arrangement (docs/plan-pdf-format.md: 未知安排留空，显示"待确认").
 */

const MAX_DESTINATION = 120
const MAX_PARTICIPANTS = 10000
const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/

/** Field-class convention copied from ``LlmConfig.tsx`` (label grid + h-10 input). */
const LABEL_CLASS = 'grid gap-1.5'
const LABEL_TEXT_CLASS = 'text-xs font-semibold text-text-secondary'
const INPUT_CLASS =
  'h-10 rounded-sm border border-border bg-bg-body px-3 text-sm outline-none ' +
  'transition-[border-color,background-color] duration-150 ease-[var(--ease-out)] ' +
  'focus:border-primary-light focus:bg-white dark:border-white/10 dark:focus:bg-[#1e1f20]'

export interface PlanDraftDialogProps {
  open: boolean
  /**
   * Park the draft: backdrop, Escape and the X all close the dialog without
   * cancelling anything (Ticket 02 AC2 — the user must be able to step away
   * and restore this conversation's draft later).
   */
  onDismiss: () => void
  /**
   * Cancel the plan flow (footer 取消) — the only exit that clears the draft
   * (Ticket 02 AC4). Falls back to parking when the parent has no separate
   * cancel action.
   */
  onCancel?: () => void
  onSubmit: (fields: V2PlanDraftFields) => void
  /** Re-opened dialog keeps what the user already typed; server fields win. */
  initial?: Partial<V2PlanDraftFields>
  /** A request is in flight: keep the dialog from submitting twice. */
  busy?: boolean
  /**
   * Ticket 02 AC3: the server-rendered draft summary for the fields already
   * stored, shown read-only above the form so an edit and its summary stay
   * together. It is the text the server produced (``plan_workflow.draft_summary``)
   * — this component never composes plan content itself, and an empty summary
   * (a draft with nothing rendered yet) renders no block.
   */
  summary?: string
}

interface FieldErrors {
  destination_or_route?: string
  activity_date?: string
  participants?: string
}

/** Deterministic client mirror of the server's minimal field validation. */
export function validatePlanDraftFields(values: {
  destination_or_route: string
  activity_date: string
  participants: string
}): FieldErrors {
  const errors: FieldErrors = {}
  const destination = values.destination_or_route.trim()
  if (!destination) errors.destination_or_route = '请填写目的地或路线'
  else if (destination.length > MAX_DESTINATION) errors.destination_or_route = `不超过 ${MAX_DESTINATION} 字`

  if (!values.activity_date) errors.activity_date = '请选择活动日期'
  else if (!ISO_DATE.test(values.activity_date)) errors.activity_date = '日期格式需为 YYYY-MM-DD'

  if (!values.participants.trim()) errors.participants = '请填写参与人数'
  else {
    const count = Number(values.participants)
    if (!Number.isInteger(count) || count < 1 || count > MAX_PARTICIPANTS) {
      errors.participants = `参与人数需为 1–${MAX_PARTICIPANTS} 的整数`
    }
  }
  return errors
}

export const PlanDraftDialog = ({ open, onDismiss, onCancel, onSubmit, initial, busy = false, summary }: PlanDraftDialogProps) => {
  const [destinationOrRoute, setDestinationOrRoute] = useState(initial?.destination_or_route ?? '')
  const [activityDate, setActivityDate] = useState(initial?.activity_date ?? '')
  const [participants, setParticipants] = useState(
    initial?.participants == null ? '' : String(initial.participants),
  )
  const [errors, setErrors] = useState<FieldErrors>({})

  // Escape mirrors the image-preview modal's handling: bound while open,
  // removed on close, and it dismisses instead of stealing the parent's key.
  useEffect(() => {
    if (!open) return
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onDismiss()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [open, onDismiss])

  // Re-seed from the server's draft whenever the dialog (re)opens, so the
  // form reflects the persisted stage rather than a stale local edit.
  useEffect(() => {
    if (!open) return
    setDestinationOrRoute(initial?.destination_or_route ?? '')
    setActivityDate(initial?.activity_date ?? '')
    setParticipants(initial?.participants == null ? '' : String(initial.participants))
    setErrors({})
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  if (!open) return null

  const handleSubmit = () => {
    const nextErrors = validatePlanDraftFields({
      destination_or_route: destinationOrRoute,
      activity_date: activityDate,
      participants,
    })
    setErrors(nextErrors)
    if (Object.keys(nextErrors).length > 0) return
    onSubmit({
      destination_or_route: destinationOrRoute.trim(),
      activity_date: activityDate,
      participants: Number(participants),
    })
  }

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 p-4 backdrop-blur-sm"
      role="dialog"
      aria-modal="true"
      aria-label="生成行程计划书"
      onClick={onDismiss}
    >
      <div
        className="w-full max-w-md rounded-xl border border-border bg-white p-5 shadow-2xl dark:border-white/10 dark:bg-[#1e1f20]"
        role="document"
        // The panel itself must not dismiss: only the backdrop does.
        onClick={event => event.stopPropagation()}
      >
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-text-primary">生成行程计划书</h2>
            <p className="mt-1 text-xs text-text-muted">
              先填三项基本信息，得到可预览的最小草稿；其余安排留空，会显示"待确认"。
            </p>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={onDismiss}
            aria-label="关闭计划书弹窗"
            title="先关掉，草稿会保留"
          >
            <X className="size-4" aria-hidden="true" />
          </Button>
        </div>

        {/* Ticket 02 AC3: the read-only summary slot ticket 01 left open.
            Server-rendered text only — this component adds no claims of its
            own, so the 待确认 / 待核验 boundary cannot be crossed here. */}
        {summary && (
          <div
            className="mb-4 rounded-md border border-border p-3 text-xs leading-relaxed whitespace-pre-wrap text-text-secondary dark:border-white/10"
            aria-label="当前草稿摘要"
          >
            {summary}
          </div>
        )}

        <form
          className="grid gap-4"
          onSubmit={event => {
            event.preventDefault()
            handleSubmit()
          }}
        >
          <label className={LABEL_CLASS}>
            <span className={LABEL_TEXT_CLASS}>目的地或路线</span>
            <input
              className={`${INPUT_CLASS} ${errors.destination_or_route ? 'border-red-500' : ''}`}
              name="destination_or_route"
              type="text"
              maxLength={MAX_DESTINATION}
              placeholder="例如：四姑娘山长坪沟"
              value={destinationOrRoute}
              onChange={event => setDestinationOrRoute(event.target.value)}
            />
            {errors.destination_or_route && (
              <span className="text-xs text-red-500" role="alert">{errors.destination_or_route}</span>
            )}
          </label>

          <label className={LABEL_CLASS}>
            <span className={LABEL_TEXT_CLASS}>活动日期</span>
            <input
              className={`${INPUT_CLASS} ${errors.activity_date ? 'border-red-500' : ''}`}
              name="activity_date"
              type="date"
              value={activityDate}
              onChange={event => setActivityDate(event.target.value)}
            />
            {errors.activity_date && (
              <span className="text-xs text-red-500" role="alert">{errors.activity_date}</span>
            )}
          </label>

          <label className={LABEL_CLASS}>
            <span className={LABEL_TEXT_CLASS}>参与人数</span>
            <input
              className={`${INPUT_CLASS} ${errors.participants ? 'border-red-500' : ''}`}
              name="participants"
              type="number"
              min={1}
              max={MAX_PARTICIPANTS}
              step={1}
              inputMode="numeric"
              placeholder="例如：12"
              value={participants}
              onChange={event => setParticipants(event.target.value)}
            />
            {errors.participants && (
              <span className="text-xs text-red-500" role="alert">{errors.participants}</span>
            )}
          </label>

          <div className="mt-1 flex justify-end gap-2">
            <Button variant="ghost" size="md" onClick={onCancel ?? onDismiss} disabled={busy}>
              取消
            </Button>
            <Button variant="accent" size="md" type="submit" disabled={busy}>
              {busy ? '提交中…' : '生成最小草稿'}
            </Button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default PlanDraftDialog
