/**
 * Inline Agent Activity Trace presenter.
 *
 * Renders the view model produced by `agentActivityModel.ts`: a collapsible
 * summary line plus the ordered activity lines. It has no data fetching, no
 * event parsing, and no approval actions — those stay in the page and the
 * existing page-level approval toast.
 */
import { useEffect, useRef, useState } from 'react'
import AlertTriangle from 'reicon-react/icons/AlertTriangle'
import ChatLine from 'reicon-react/icons/ChatLine'
import Check from 'reicon-react/icons/Check'
import Clock from 'reicon-react/icons/Clock'
import Document from 'reicon-react/icons/Document'
import Lightbulb from 'reicon-react/icons/Lightbulb'
import Loader from 'reicon-react/icons/Loader'
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
  running: Loader,
  done: Check,
  error: AlertTriangle,
  pending: Clock,
}

function summaryStatus(model: AgentActivityModel): TraceStatus {
  if (model.lines.some(line => line.status === 'error')) return 'error'
  if (model.lines.some(line => line.status === 'pending')) return 'pending'
  if (model.lines.some(line => line.status === 'running')) return 'running'
  return 'done'
}

function lineIcon(line: ActivityLine) {
  if (line.status === 'error') return AlertTriangle
  if (line.status === 'pending') return Clock
  if (line.status === 'running') return Loader

  switch (line.kind) {
    case 'tool':
    case 'approval':
      return Check
    case 'artifact':
      return Document
    case 'review':
      return AlertTriangle
    case 'thought':
      return Lightbulb
    case 'answer':
      return ChatLine
    default:
      return Check
  }
}

function iconClassName(status: TraceStatus) {
  return `h-3.5 w-3.5${status === 'running' ? ' animate-spin motion-reduce:animate-none' : ''}`
}

export function AgentActivityTrace({
  model,
  className,
}: {
  model: AgentActivityModel
  className?: string
}) {
  const [expanded, setExpanded] = useState(model.defaultExpanded)
  const lastDefaultExpanded = useRef(model.defaultExpanded)

  // A user toggle wins until the model asks for a different default (for
  // example, a clean completion collapsing the trace).
  useEffect(() => {
    if (lastDefaultExpanded.current === model.defaultExpanded) return
    lastDefaultExpanded.current = model.defaultExpanded
    setExpanded(model.defaultExpanded)
  }, [model.defaultExpanded])

  if (!model.visible) return null

  const status = summaryStatus(model)
  const SummaryIcon = SUMMARY_ICONS[status]
  const summaryLineStatus = status === 'done' ? undefined : status

  return (
    <div
      data-slot="agent-activity"
      data-status={status}
      className={`w-full min-w-0${className ? ` ${className}` : ''}`}
    >
      <TraceCollapsible open={expanded} onOpenChange={setExpanded}>
        <TraceCollapsibleTrigger>
          <TraceLine status={summaryLineStatus}>
            <TraceIcon>
              <SummaryIcon className={iconClassName(status)} />
            </TraceIcon>
            <TraceTitle>{model.summary}</TraceTitle>
          </TraceLine>
        </TraceCollapsibleTrigger>
        <TraceCollapsibleContent>
          <Trace className="mt-1.5 ml-[7px] min-w-0 border-l border-border/70 pl-3">
            {model.lines.map(line => {
              const Icon = lineIcon(line)
              return (
                <TraceLine
                  key={line.id}
                  status={line.status === 'done' ? undefined : line.status}
                  className="py-0.5"
                >
                  <TraceIcon>
                    <Icon className={iconClassName(line.status)} />
                  </TraceIcon>
                  <TraceTitle>{line.title}</TraceTitle>
                  {line.description && (
                    <TraceDescription>{line.description}</TraceDescription>
                  )}
                </TraceLine>
              )
            })}
          </Trace>
        </TraceCollapsibleContent>
      </TraceCollapsible>
      <span className="sr-only" role="status" aria-live="polite">
        当前状态：{model.summary}
      </span>
    </div>
  )
}

export default AgentActivityTrace
