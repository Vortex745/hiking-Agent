/**
 * Inline Agent Activity Trace presenter.
 *
 * Renders the view model produced by `agentActivityModel.ts`: a collapsible
 * summary line plus the ordered activity lines. It has no data fetching, no
 * event parsing, and no approval actions — those stay in the page and the
 * existing page-level approval toast.
 */
import { useEffect, useRef, useState } from 'react'
import { Check, CircleHelp, Clock, FileText, Lightbulb, LoaderCircle, MessageSquareText, TriangleAlert } from 'lucide-react'
import {
  Trace,
  TraceCollapsible,
  TraceCollapsibleContent,
  TraceCollapsibleTrigger,
  TraceDescription,
  TraceIcon,
  TraceLine,
  TraceTitle,
} from '../ui/trace'
import type {
  ActivityLine,
  ActivityStatus,
  AgentActivityModel,
} from './agentActivityModel'

type TraceStatus = ActivityStatus

const SUMMARY_ICONS = {
  // The summary never spins: only the step that is actually executing carries
  // the rotating indicator, so two spinners cannot compete in one trace.
  running: Lightbulb,
  done: Check,
  error: TriangleAlert,
  pending: Clock,
  unknown: CircleHelp,
}

const ICON_CLASS = 'h-3.5 w-3.5'

function lineIcon(line: ActivityLine) {
  if (line.status === 'error') return TriangleAlert
  if (line.status === 'unknown') return CircleHelp
  if (line.status === 'pending') return Clock
  if (line.status === 'running') return LoaderCircle

  switch (line.kind) {
    case 'tool':
    case 'approval':
      return Check
    case 'artifact':
      return FileText
    case 'review':
      return TriangleAlert
    case 'thought':
      return Lightbulb
    case 'answer':
      return MessageSquareText
    default:
      return Check
  }
}

function iconClassName(status: TraceStatus) {
  return `${ICON_CLASS}${status === 'running' ? ' animate-spin motion-reduce:animate-none' : ''}`
}

export function AgentActivityTrace({
  model,
  className,
}: {
  model: AgentActivityModel
  className?: string
}) {
  const [expanded, setExpanded] = useState(model.defaultExpanded)
  const lastStatus = useRef(model.status)

  // A user toggle wins until the summary status changes (for example, a
  // clean completion or a failed step). Keying on the status, not the
  // default-expanded boolean, lets a failed run re-expand even when the
  // boolean was already true.
  useEffect(() => {
    if (lastStatus.current === model.status) return
    lastStatus.current = model.status
    setExpanded(model.defaultExpanded)
  }, [model.status, model.defaultExpanded])

  if (!model.visible) return null

  const status = model.status
  const SummaryIcon = SUMMARY_ICONS[status]
  const summaryLineStatus = status === 'done' || status === 'unknown' ? undefined : status

  return (
    <div
      data-slot="agent-activity"
      data-status={status}
      className={`w-full min-w-0${className ? ` ${className}` : ''}`}
    >
      <div
        data-slot="agent-activity-block"
        className="w-fit max-w-full min-w-0 rounded-2xl bg-[#f3f0ed] px-3 py-2 dark:bg-white/[0.06]"
      >
        <TraceCollapsible open={expanded} onOpenChange={setExpanded}>
          <TraceCollapsibleTrigger>
            <TraceLine status={summaryLineStatus}>
              <TraceIcon>
                <SummaryIcon className={ICON_CLASS} />
              </TraceIcon>
              <TraceTitle>{model.summary}</TraceTitle>
            </TraceLine>
          </TraceCollapsibleTrigger>
          <TraceCollapsibleContent>
            <Trace className="mt-1.5 ml-[7px] min-w-0 border-l border-border/70 pl-3">
              {model.lines.map(line => {
                const Icon = lineIcon(line)
                const lineStatus =
                  line.status === 'done' || line.status === 'unknown'
                    ? undefined
                    : line.status
                return (
                  <div key={line.id} className="flex flex-col gap-1 py-0.5">
                    <TraceLine status={lineStatus}>
                      <TraceIcon>
                        <Icon className={iconClassName(line.status)} />
                      </TraceIcon>
                      <TraceTitle>{line.title}</TraceTitle>
                      {line.description && (
                        <TraceDescription className="max-w-[55%] truncate">
                          {line.description}
                        </TraceDescription>
                      )}
                    </TraceLine>
                    {line.detail && (
                      <div
                        data-slot="trace-detail"
                        className="max-h-64 overflow-y-auto whitespace-pre-wrap break-words pl-5 pr-1 text-[13px] leading-relaxed text-text-secondary"
                      >
                        {line.detail}
                      </div>
                    )}
                  </div>
                )
              })}
            </Trace>
          </TraceCollapsibleContent>
        </TraceCollapsible>
      </div>
      <span className="sr-only" role="status" aria-live="polite">
        当前状态：{model.summary}
      </span>
    </div>
  )
}

export default AgentActivityTrace
