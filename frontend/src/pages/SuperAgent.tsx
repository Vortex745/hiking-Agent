import { useState, useRef, useEffect, useCallback, useMemo, type Dispatch, type ReactNode, type SetStateAction } from 'react'
import { API } from '../api/config'
import {
  createV2StreamConnection,
  cancelV2Run,
  fetchV2Snapshot,
  fetchV2Events,
  submitV2Approval,
  buildV2ResumeUrlFromDecision,
  submitV2MemoryDecision,
  planV2Recovery,
  V2_TERMINAL_RUN_STATUSES,
  type V2EventEnvelope,
  type V2RunSnapshot,
  type V2ApprovalRequest,
  type V2ApprovalDecisionResponse,
  type V2MemoryCandidate,
} from '../api/chatV2'
import {
  reduceV2Event,
  reduceV2Events,
  initialV2State,
  isV2Terminal,
  isV2Paused,
  reconcileV2Snapshot,
  assistantTextFromSnapshot,
  buildRecoveredTrace,
  readArtifact as readArtifactFromPayload,
  type V2ConversationState,
} from '../api/v2Reducer'
import { useExternalStoreRuntime, AssistantRuntimeProvider, type ExternalStoreAdapter, type ThreadMessage } from "@assistant-ui/react"
import { GeminiThread } from '../components/assistant-ui/gemini/GeminiThread'
import { GeminiSidebar } from '../components/assistant-ui/gemini/GeminiSidebar'
import { CircleCheck, Clipboard, Database, Map, Plug, Settings2, SlidersHorizontal, TriangleAlert } from 'lucide-react'
import { DropdownMenu } from '../components/ui/dropdown-menu'
import { MemoryDialog } from '../components/MemoryDialog'
import { McpDialog } from '../components/McpDialog'
import KnowledgeSourcesDialog from './KnowledgeSources'
import LlmConfigDialog from './LlmConfig'

interface Message {
  role: 'user' | 'assistant'
  content: string
  time?: string
  isStreaming?: boolean
  traceEvents?: AgentTraceEvent[]
  artifacts?: AgentArtifact[]
}

interface ChatSession {
  id: string
  chatId: string
  title: string
  date: string
  messages: Message[]
}

type AgentScenario = 'route_plan' | 'gear_check' | 'risk_assessment' | 'report_export'
type AgentTraceEvent = {
  type: string
  content: string
  metadata?: Record<string, unknown>
}
/** 从当前会话最近的 focus_map 工具调用中取唯一景点；没有则返回 null。 */
function extractMapFocus(messages: Message[]): { name: string } | null {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const events = messages[i].traceEvents
    if (!events?.length) continue
    for (let j = events.length - 1; j >= 0; j -= 1) {
      const event = events[j]
      const metadata = event.metadata
      if (!metadata || metadata.tool !== 'focus_map') continue
      if (event.type === 'tool_result') {
        const prefix = '地图定位：'
        const content = event.content.trim()
        if (content.startsWith(prefix)) {
          const name = content.slice(prefix.length).trim()
          if (name) return { name }
        }
        continue
      }
      if (event.type !== 'tool_call') continue
      const args = metadata.args
      if (!args || typeof args !== 'object') continue
      const name = (args as Record<string, unknown>).name
      if (typeof name === 'string' && name.trim()) return { name: name.trim() }
    }
  }
  return null
}

type TraceAppendOptions = {
  /** Replacement content; a function receives the current last message. */
  content?: string | ((last: Message) => string)
  /** Force the streaming flag. Omit to preserve the message's current value. */
  isStreaming?: boolean
}

/**
 * Appends a trace event to the last assistant message. All SSE branches use
 * this helper; most force `isStreaming: true`, while a replayed
 * `approval.resolved` omits it so a finished message stays finished.
 */
function appendTraceEvent(
  setMessages: Dispatch<SetStateAction<Message[]>>,
  traceEvent: AgentTraceEvent,
  options: TraceAppendOptions = {},
) {
  const { content, isStreaming } = options
  setMessages(prev => {
    const updated = [...prev]
    const last = updated[updated.length - 1]
    if (!last || last.role !== 'assistant') return prev
    const next: Message = {
      ...last,
      traceEvents: [...(last.traceEvents || []), traceEvent],
    }
    if (content !== undefined) {
      next.content = typeof content === 'function' ? content(last) : content
    }
    if (isStreaming !== undefined) next.isStreaming = isStreaming
    updated[updated.length - 1] = next
    return updated
  })
}
// A02: v2 artifact shape (payload of artifact.available). Download goes
// through /api/v2/artifacts/{artifact_id} — no v1 download_url exists.
type AgentArtifact = {
  artifact_id: string
  public_filename: string
  mime_type: string
  size_bytes: number
  sha256?: string
  expires_at?: string
}
type BrowserLocation = {
  latitude: number
  longitude: number
  accuracy?: number
  source: 'browser'
}

function shouldRequestCurrentLocation(text: string) {
  return /天气|适合|能去|可以去|徒步吗|去徒步|附近|周边|当前位置|我这里|我这边/.test(text)
}

function isAffirmativeRouteReply(text: string) {
  return /^(需要|要|好|好的|可以|行|安排|推荐|继续)$/.test(text.trim())
}

function shouldRequestRouteFollowupLocation(text: string, messages: Message[]) {
  if (!isAffirmativeRouteReply(text)) return false
  const lastAssistant = [...messages].reverse().find(message => message.role === 'assistant')
  return Boolean(lastAssistant?.content.includes('要不要我继续给你推荐附近的户外徒步路线'))
}

const quickTags = [
  { icon: Clipboard, text: '生成行程', scenario: 'route_plan' as AgentScenario },
  { icon: CircleCheck, text: '检查装备', scenario: 'gear_check' as AgentScenario },
  { icon: TriangleAlert, text: '风险提醒', scenario: 'risk_assessment' as AgentScenario },
  { icon: Map, text: '推荐路线', scenario: 'route_plan' as AgentScenario },
]

const STORAGE_KEY = 'ai-hiking-agent-chat'
const SESSIONS_KEY = 'ai-hiking-agent-sessions'
const CHAT_ID_KEY = 'ai-hiking-agent-chat-id'
const ACTIVE_SESSION_KEY = 'ai-hiking-agent-active-session'
const SENSITIVE_LOCATION_FIELDS = new Set(['latitude', 'longitude', 'accuracy'])
const USER_CANCELLED_MESSAGE = '用户主动取消回答，请补充内容或点击重试。'

// A06: recovery polling. A running run cannot be re-attached (a duplicate
// POST /chat/sse with the same request_id replays stored events and closes
// without a terminal), so recovery replays once and then polls the
// snapshot until the run reaches a terminal or pauses.
const RECOVERY_POLL_INTERVAL_MS = 3_000
const RECOVERY_POLL_TIMEOUT_MS = 10 * 60 * 1_000

// A06 review fix: render the reducer's reviewNotices as the same
// "[注意] ..." lines the live stream appends, so replays/polls preserve
// them instead of erasing the human-review context.
const v2NoticeSuffix = (notices: string[]): string =>
  notices.map(n => `

[注意] ${n}`).join('')

// Ticket 08: persisted v2 run state for refresh recovery (spec AC3).
// We store the active run_id + sequence cursor + conversation_id so a
// page refresh can call /snapshot + /events to resume without repeating
// user messages or already-displayed text.
//
// Ticket 11: ``pending_approval`` mirrors the reducer's
// ``pendingApproval`` so a refresh while paused re-shows the prompt
// without round-tripping through /events first. ``v2_status`` lets the
// recovery effect skip the snapshot fetch entirely when the run is
// already terminal locally (avoids a 404 race when the server reaped
// the run row).
interface V2PersistedRun {
  run_id: string
  request_id: string
  conversation_id: string
  sequence_cursor: number
  user_message: string
  // Ticket 11: present only when the run is paused awaiting approval.
  pending_approval?: V2ApprovalRequest | null
  // Ticket 11: last-known run status so recovery knows whether to
  // re-attach a stream, re-show the approval prompt, or do nothing.
  // BCDEF ticket 11: ``outcome_unknown`` / ``admission_failed`` are
  // first-class terminal statuses — recovery trusts them locally and
  // never re-executes (they are NON-continuable; only ``paused`` resumes).
  v2_status?: 'running' | 'paused' | 'done' | 'error' | 'cancelled' | 'outcome_unknown' | 'admission_failed'
}
const V2_RUN_KEY = 'ai-hiking-agent-v2-active-run'
function loadV2PersistedRun(): V2PersistedRun | null {
  try {
    const raw = localStorage.getItem(V2_RUN_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as Partial<V2PersistedRun>
    if (
      typeof parsed.run_id !== 'string' ||
      typeof parsed.request_id !== 'string' ||
      typeof parsed.conversation_id !== 'string' ||
      typeof parsed.sequence_cursor !== 'number' ||
      typeof parsed.user_message !== 'string'
    ) {
      return null
    }
    return parsed as V2PersistedRun
  } catch { return null }
}
function saveV2PersistedRun(run: V2PersistedRun | null): void {
  try {
    if (run === null) localStorage.removeItem(V2_RUN_KEY)
    else localStorage.setItem(V2_RUN_KEY, JSON.stringify(run))
  } catch { /* ignore */ }
}

function createChatId() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return `chat-${crypto.randomUUID()}`
  }
  return `chat-${Date.now()}-${Math.random().toString(36).slice(2)}`
}

function getOrCreateChatId() {
  try {
    const saved = localStorage.getItem(CHAT_ID_KEY)
    if (saved) return saved

    const next = createChatId()
    localStorage.setItem(CHAT_ID_KEY, next)
    return next
  } catch {
    return createChatId()
  }
}

function sanitizeStoredValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(sanitizeStoredValue)
  }
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value)
        .filter(([key]) => !SENSITIVE_LOCATION_FIELDS.has(key))
        .map(([key, item]) => [key, sanitizeStoredValue(item)]),
    )
  }
  return value
}

function serializeMessages(messages: Message[]) {
  return JSON.stringify(sanitizeStoredValue(messages))
}

function sanitizeSessionForStorage(session: ChatSession): ChatSession {
  return {
    ...session,
    messages: sanitizeStoredValue(session.messages) as Message[],
  }
}

function generateId() {
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`
}

function getTodayDate() {
  const now = new Date()
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`
}

function loadSessions(): ChatSession[] {
  try {
    const saved = localStorage.getItem(SESSIONS_KEY)
    const parsed = saved ? JSON.parse(saved) : []
    if (!Array.isArray(parsed)) return []
    return parsed.map((session: Partial<ChatSession>) => ({
      id: session.id || generateId(),
      chatId: session.chatId || createChatId(),
      title: session.title || '新对话',
      date: session.date || getTodayDate(),
      messages: Array.isArray(session.messages) ? session.messages : [],
    }))
  } catch { return [] }
}

function saveSessions(sessions: ChatSession[]) {
  try { localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions.map(sanitizeSessionForStorage))) } catch { /* ignore */ }
}

function getBrowserLocation(timeout = 3000): Promise<BrowserLocation | null> {
  if (typeof navigator === 'undefined' || !navigator.geolocation) {
    return Promise.resolve(null)
  }

  return new Promise(resolve => {
    navigator.geolocation.getCurrentPosition(
      position => {
        resolve({
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
          accuracy: position.coords.accuracy,
          source: 'browser',
        })
      },
      () => resolve(null),
      {
        enableHighAccuracy: false,
        maximumAge: 10 * 60 * 1000,
        timeout,
      },
    )
  })
}

const convertMessage = (msg: Message, index: number): ThreadMessage => {
  const content: any[] = [{ type: "text", text: msg.content || (msg.isStreaming ? "" : " ") }];

  if (msg.traceEvents && msg.traceEvents.length > 0) {
    content.push({
      type: "tool-call",
      toolCallId: `trace-${index}`,
      toolName: "traceEvents",
      args: msg.traceEvents
    });
  }
  if (msg.artifacts && msg.artifacts.length > 0) {
    content.push({
      type: "tool-call",
      toolCallId: `artifact-${index}`,
      toolName: "artifacts",
      args: msg.artifacts
    });
  }

  // A09: aggregate retrieval evidence from tool_result payloads into a
  // message-level 检索来源 part (dedup by source+preview).
  const evidenceSeen = new Set<string>();
  const evidenceChunks: Array<{ title?: string; source: string; preview: string }> = [];
  for (const evt of msg.traceEvents || []) {
    const ev = evt.metadata?.evidence as { has_chunks?: boolean; sources?: unknown; chunks?: Array<Record<string, unknown>> } | undefined;
    if (!ev || !ev.has_chunks) continue;
    if (Array.isArray(ev.chunks)) {
      for (const chunk of ev.chunks) {
        if (!chunk || typeof chunk !== 'object') continue;
        const source = typeof chunk.source === 'string' ? chunk.source : '';
        const preview = typeof chunk.preview === 'string' ? chunk.preview : '';
        if (!source && !preview) continue;
        const key = `${source}|${preview}`;
        if (evidenceSeen.has(key)) continue;
        evidenceSeen.add(key);
        evidenceChunks.push({
          title: typeof chunk.title === 'string' ? chunk.title : undefined,
          source,
          preview,
        });
      }
    }
    // Review fix (A09 AC): the payload's top-level `sources` summary list
    // must render too — append docs that no chunk already covered.
    if (Array.isArray(ev.sources)) {
      for (const src of ev.sources) {
        if (typeof src !== 'string' || !src) continue;
        const key = `${src}|`;
        if (evidenceSeen.has(key) || evidenceChunks.some(c => c.source === src)) continue;
        evidenceSeen.add(key);
        evidenceChunks.push({ source: src, preview: '' });
      }
    }
  }
  if (evidenceChunks.length > 0) {
    content.push({
      type: "tool-call",
      toolCallId: `evidence-${index}`,
      toolName: "evidence",
      args: evidenceChunks
    });
  }

  return {
    id: `msg-${index}`,
    role: msg.role,
    content: content,
    status: (msg.isStreaming ? { type: "running" } : { type: "complete" }),
    createdAt: new Date(),
    metadata: {},
  } as unknown as ThreadMessage;
};

// The external-store hook updates its retained runtime in an effect. Reusing
// it for a shorter conversation notifies the old index-based message readers
// before they unmount, which can throw an out-of-bounds error. Key this owner
// by chat identity so the runtime and all of its readers change together.
function SessionRuntimeProvider({ adapter, children }: {
  adapter: ExternalStoreAdapter<ThreadMessage>
  children: ReactNode
}) {
  const runtime = useExternalStoreRuntime(adapter)
  return <AssistantRuntimeProvider runtime={runtime}>{children}</AssistantRuntimeProvider>
}

function SuperAgent() {
  const [messages, setMessages] = useState<Message[]>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY)
      return saved ? JSON.parse(saved) : []
    } catch { return [] }
  })
  const mapFocus = useMemo(() => extractMapFocus(messages), [messages])
  const [sessions, setSessions] = useState<ChatSession[]>(loadSessions)
  const [activeSessionId, setActiveSessionId] = useState<string | null>(() => {
    try {
      const saved = localStorage.getItem(ACTIVE_SESSION_KEY)
      if (saved) return saved
      const savedMessages = localStorage.getItem(STORAGE_KEY)
      if (savedMessages) {
        const msgs = JSON.parse(savedMessages)
        if (msgs.length > 0) return generateId()
      }
      return null
    } catch { return null }
  })
  // 会话侧栏（assistant-ui gemini 示例的停靠左栏）：桌面端折叠/展开，
  // 移动端为左上角菜单按钮 + 抽屉。桌面默认展开列表。
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false)

  const [input, setInput] = useState('')
  const [selectedScenario, setSelectedScenario] = useState<AgentScenario | null>(null)
  const [isSending, setIsSending] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connected' | 'error'>('disconnected')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const cleanupRef = useRef<(() => void) | null>(null)
  const chatIdRef = useRef(getOrCreateChatId())
  const lastLocationRef = useRef<BrowserLocation | null>(null)

  // Chat v2 是唯一 chat 链路（v1 已移除）。The v2 run ref tracks the
  // active run_id + sequence cursor so refresh recovery can call
  // /snapshot + /events to resume without re-sending the user message
  // or re-displaying already-shown text (spec AC3).
  //
  // Ticket 11: ``v2ApprovalState`` tracks the in-flight decision so the
  // UI can render "处理中" while the POST /approvals/{id}/decision
  // request is in flight, and "已解决" / "恢复失败" once the resume
  // stream opens or fails. It is reset to 'idle' on the next
  // ``approval.required`` event. ``pendingApproval`` mirrors the
  // reducer's ``pendingApproval`` as React state so the approve/reject
  // prompt re-renders when a run pauses.
  const [configOpen, setConfigOpen] = useState(false)
  const [memoryDialogOpen, setMemoryDialogOpen] = useState(false)
  const [mcpDialogOpen, setMcpDialogOpen] = useState(false)
  const [knowledgeSourcesOpen, setKnowledgeSourcesOpen] = useState(false)
  const [memoryRefreshKey, setMemoryRefreshKey] = useState(0)
  const suppressMenuRefocus = useRef(false)
  const settingsTriggerRef = useRef<HTMLButtonElement>(null)
  const [pendingMemoryConsent, setPendingMemoryConsent] = useState<V2MemoryCandidate | null>(null)
  const [memoryDecisionPending, setMemoryDecisionPending] = useState(false)
  const [memoryDecisionError, setMemoryDecisionError] = useState('')
  const [pendingApproval, setPendingApproval] = useState<V2ApprovalRequest | null>(null)
  const v2RunRef = useRef<V2PersistedRun | null>(loadV2PersistedRun())
  const v2StateRef = useRef<V2ConversationState>(initialV2State)
  const [v2ApprovalState, setV2ApprovalState] = useState<
    'idle' | 'submitting' | 'resuming' | 'resolved' | 'error'
  >('idle')

  const closeLlmConfig = useCallback(() => {
    setConfigOpen(false)
    window.requestAnimationFrame(() => settingsTriggerRef.current?.focus())
  }, [])

  const closeMemoryDialog = useCallback(() => {
    setMemoryDialogOpen(false)
    window.requestAnimationFrame(() => settingsTriggerRef.current?.focus())
  }, [])

  const closeMcpDialog = useCallback(() => {
    setMcpDialogOpen(false)
    window.requestAnimationFrame(() => settingsTriggerRef.current?.focus())
  }, [])

  const handleMemoryDecision = useCallback(async (decision: 'approve' | 'reject') => {
    if (!pendingMemoryConsent || memoryDecisionPending) return
    setMemoryDecisionPending(true)
    setMemoryDecisionError('')
    try {
      await submitV2MemoryDecision(
        pendingMemoryConsent.candidate_id,
        decision,
        `memory-decision-${Date.now()}`,
      )
      setPendingMemoryConsent(null)
      v2StateRef.current = { ...v2StateRef.current, pendingMemoryConsent: null }
      setMemoryRefreshKey(key => key + 1)
    } catch (error) {
      setMemoryDecisionError(error instanceof Error ? error.message : '记忆决策失败')
    } finally {
      setMemoryDecisionPending(false)
    }
  }, [memoryDecisionPending, pendingMemoryConsent])


  // Ticket 08: v2 event handlers (spec AC2). Map v2 wire events to stable
  // UI state. These are defined as useCallbacks before the recovery effect
  // so the effect can reference them without forward-reference issues.
  // Store transport deltas immediately; MarkdownText owns visual pacing.
  const handleV2Event = useCallback((event: V2EventEnvelope) => {
    const prevCursor = v2StateRef.current.sequenceCursor
    const next = reduceV2Event(v2StateRef.current, event)
    v2StateRef.current = next

    // Spec decision #26: the server embeds run_id in every envelope. The
    // frontend learns run_id from the first event (it is generated by
    // create_run_idempotent and is not known at request time). Persist
    // it so refresh recovery can call /snapshot/{run_id} etc.
    if (event.run_id && v2RunRef.current && v2RunRef.current.run_id !== event.run_id) {
      v2RunRef.current = { ...v2RunRef.current, run_id: event.run_id }
      saveV2PersistedRun(v2RunRef.current)
    }

    // Persist the new cursor so refresh recovery resumes from here.
    // Ticket 11: also persist ``pending_approval`` and ``v2_status`` so
    // a refresh while paused re-shows the prompt without round-tripping
    // through /events first.
    //
    // BCDEF ticket 11 (AC6): ``run.paused`` must NOT advance the persisted
    // cursor. The pause only becomes durable when the control row reaches
    // ``paused`` AFTER the frame was already streamed — a refresh landing
    // inside that window would replay from a cursor past the pause and
    // lose the approval prompt. Holding the cursor one frame back keeps
    // the pause replayable (reducing the replay is idempotent), and the
    // prompt itself is mirrored into ``pending_approval`` + restored from
    // snapshot ``pending_approvals`` as belt-and-braces.
    if (v2RunRef.current) {
      v2RunRef.current = {
        ...v2RunRef.current,
        sequence_cursor: event.type === 'run.paused' ? prevCursor : next.sequenceCursor,
        pending_approval: next.pendingApproval,
        v2_status: next.status === 'idle' ? undefined : next.status,
      }
      saveV2PersistedRun(v2RunRef.current)
    }

    // Ticket 11: reset the approval UI state when a new approval is
    // requested. The decision lifecycle (submitting → resuming →
    // resolved/error) is driven by handleV2Approval below.
    if (event.type === 'approval.required') {
      setV2ApprovalState('idle')
      setPendingApproval(next.pendingApproval)
    }
    // Ticket 11: ``approval.resolved`` arrives on the resume stream —
    // the decision has been recorded server-side and the run is
    // continuing. Mark the UI as resolved so the approve/reject buttons
    // dismiss gracefully while the next events arrive.
    if (event.type === 'approval.resolved') {
      setV2ApprovalState('resolved')
      setPendingApproval(null)

      // Forward the resolution into the message's trace events so the inline
      // activity trace settles its approval wait line. Preserve the current
      // streaming flag: a replayed resolution must not reopen a finished run.
      const traceEvent: AgentTraceEvent = {
        type: 'approval_resolved',
        content: '',
        metadata: event.payload as Record<string, unknown>,
      }
      appendTraceEvent(setMessages, traceEvent)
    }
    if (event.type === 'memory.consent_required') {
      setPendingMemoryConsent(next.pendingMemoryConsent)
    }

    if (event.type === 'progress' && event.payload.stage === 'model') {
      // Agent process tracking ticket 03: a model round boundary rides the
      // existing ``progress`` type. It becomes its own timeline line; the
      // start carries the round label, the finish carries the duration.
      const traceEvent: AgentTraceEvent = {
        type: 'model_round',
        content:
          typeof event.payload.message === 'string' ? event.payload.message : '',
        metadata: event.payload as Record<string, unknown>,
      }
      appendTraceEvent(setMessages, traceEvent, { isStreaming: true })
    }
    if (event.type === 'text') {
      const delta = typeof event.payload.content === 'string' ? event.payload.content : ''
      if (!delta) return
      setMessages(prev => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        if (last && last.role === 'assistant') {
          updated[updated.length - 1] = { ...last, content: last.content + delta, isStreaming: true }
        }
        return updated
      })
    } else if (event.type === 'thought') {
      // Model reasoning the kernel's public-text filter removed. Providers
      // without reasoning emit no such event, so the trace simply stays
      // without a thought line.
      const content = typeof event.payload.content === 'string' ? event.payload.content : ''
      if (!content) return
      const traceEvent: AgentTraceEvent = {
        type: 'thought',
        content,
        metadata: event.payload as Record<string, unknown>,
      }
      appendTraceEvent(setMessages, traceEvent, { isStreaming: true })
    } else if (event.type === 'tool_call' || event.type === 'tool_result') {
      const traceEvent: AgentTraceEvent = {
        type: event.type,
        content: typeof event.payload.content === 'string' ? event.payload.content : '',
        metadata: event.payload as Record<string, unknown>,
      }
      appendTraceEvent(setMessages, traceEvent, { isStreaming: true })
    } else if (event.type === 'artifact_candidate') {
      // A02: candidate artifact produced, host-side export pending.
      const traceEvent: AgentTraceEvent = {
        type: 'artifact_progress',
        content:
          typeof event.payload.message === 'string' && event.payload.message
            ? event.payload.message
            : '正在生成产物…',
        metadata: event.payload as Record<string, unknown>,
      }
      appendTraceEvent(setMessages, traceEvent, { isStreaming: true })
    } else if (event.type === 'artifact.available') {
      // A02: verified artifact ready — attach the download card to the
      // active assistant message. The payload is the public shape only.
      const artifact = readArtifactFromPayload(event.payload)
      if (!artifact) return
      setMessages(prev => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        if (last && last.role === 'assistant') {
          updated[updated.length - 1] = {
            ...last,
            artifacts: [...(last.artifacts || []), artifact],
            isStreaming: true,
          }
        }
        return updated
      })
    } else if (event.type === 'progress.gap') {
      // BCDEF ticket 11: droppable progress left a hole — the server's
      // replay synthesizes this envelope (the stream loop synthesizes it
      // locally for live jumps). Show the gap; never treat it as stalled
      // progress or completion, and never trigger a model call.
      const notice = next.reviewNotices[next.reviewNotices.length - 1]
        || `部分进度更新丢失（序列 ${String(event.payload.from_sequence ?? '?')} → ${String(event.payload.to_sequence ?? '?')}）`
      const traceEvent: AgentTraceEvent = {
        type: 'review_notice',
        content: notice,
        metadata: event.payload as Record<string, unknown>,
      }
      appendTraceEvent(setMessages, traceEvent, {
        content: last => `${last.content}${last.content.trim() ? '\n\n' : ''}[注意] ${notice}`,
        isStreaming: true,
      })
    } else if (event.type === 'effect.outcome_unknown') {
      // A02: side effect finished with unknown outcome — the server
      // contract emits this before the terminal error. Surface the
      // human-review notice in the main flow (same pattern as the
      // error terminal) plus the trace panel.
      const notice =
        (typeof event.payload.message === 'string' && event.payload.message) ||
        (typeof event.payload.code === 'string' && event.payload.code) ||
        '部分操作结果未知，需要人工核查'
      const traceEvent: AgentTraceEvent = {
        type: 'review_notice',
        content: notice,
        metadata: event.payload as Record<string, unknown>,
      }
      appendTraceEvent(setMessages, traceEvent, {
        content: last => `${last.content}${last.content.trim() ? '\n\n' : ''}[注意] ${notice}`,
        isStreaming: true,
      })
    } else if (event.type === 'approval.required') {
      // Surface the approval request as a trace event so the lifecycle
      // panel shows why the run paused. The actual approve/reject
      // buttons render from ``v2StateRef.current.pendingApproval``.
      const traceEvent: AgentTraceEvent = {
        type: 'approval_required',
        content: '',
        metadata: event.payload as Record<string, unknown>,
      }
      appendTraceEvent(setMessages, traceEvent, { isStreaming: true })
    }
    // `progress` events update the v2 state (already done above) but do
    // not touch messages — the UI's thinking indicator reads from
    // v2StateRef via the status label. `run.paused` is handled by the
    // stream onTerminal callback below.
  }, [])

  // A07: conversation ownership follows the subject cookie. After a
  // cookie loss the old chatId is dead — rotate to a fresh
  // conversation. Review fix: also sync the active session entry so
  // a refresh before the next message does not restore the dead
  // chatId from the session list.
  const rotateChatIdAfterForbidden = useCallback(() => {
    const next = createChatId()
    localStorage.setItem(CHAT_ID_KEY, next)
    chatIdRef.current = next
    if (activeSessionId) {
      setSessions(prev =>
        prev.map(s => s.id === activeSessionId ? { ...s, chatId: next } : s),
      )
    }
  }, [activeSessionId])

  const handleV2Terminal = useCallback((event: V2EventEnvelope) => {
    // Ticket 11: ``run.paused`` is stream-terminal but NOT run-terminal.
    // The reducer has already transitioned status to ``paused``. We
    // deliberately keep ``isSending=true`` and ``v2RunRef`` intact so
    // the UI stays in "awaiting approval" mode and the user can click
    // approve/reject. The SSE connection is naturally torn down because
    // the server closed the stream; we only null ``cleanupRef`` so a
    // subsequent cancel click does not double-cancel a dead stream.
    if (event.type === 'run.paused') {
      cleanupRef.current = null
      return
    }

    // Run-terminal events (done/error/cancelled): finalize the UI and
    // clear the persisted run (spec AC3 + AC4).
    v2RunRef.current = null
    saveV2PersistedRun(null)
    setV2ApprovalState('idle')
    setPendingApproval(null)

    setMessages(prev => {
      const updated = [...prev]
      const last = updated[updated.length - 1]
      if (!last || last.role !== 'assistant') return updated

      if (event.type === 'cancelled') {
        // Spec AC4: show "cancelled", not a generic network error.
        const notice = USER_CANCELLED_MESSAGE
        const separator = last.content.trim() ? '\n\n' : ''
        return [
          ...updated.slice(0, -1),
          { ...last, content: `${last.content}${separator}${notice}`, isStreaming: false },
        ]
      }
      if (event.type === 'error') {
        const code = typeof event.payload.code === 'string' ? event.payload.code : ''
        if (code === 'conversation_forbidden') {
          // A07 review fix: rotate OUTSIDE the messages updater —
          // React updaters must stay side-effect free (StrictMode
          // may double-invoke them). Old turns stay visible
          // read-only in history.
          rotateChatIdAfterForbidden()
          const separator = last.content.trim() ? '\n\n' : ''
          return [
            ...updated.slice(0, -1),
            {
              ...last,
              content: `${last.content}${separator}[提示] 原会话已失效，已为你开启新对话，请重新发送消息。`,
              isStreaming: false,
            },
          ]
        }
        const message =
          (typeof event.payload.message === 'string' && event.payload.message) ||
          (typeof event.payload.code === 'string' && event.payload.code) ||
          '运行出错'
        const separator = last.content.trim() ? '\n\n' : ''
        // BCDEF ticket 11: an unknown external-effect outcome is a
        // distinct terminal, not a generic failure. Render it apart (the
        // reducer already set status ``outcome_unknown`` — sticky,
        // non-continuable, never auto-retried) so the user waits for
        // background reconciliation instead of being prompted to retry.
        const outcomeUnknown =
          code === 'effect_outcome_unknown' ||
          code === 'outcome_unknown' ||
          v2StateRef.current.status === 'outcome_unknown'
        return [
          ...updated.slice(0, -1),
          {
            ...last,
            content: `${last.content}${separator}${outcomeUnknown ? '[结果未知] ' : '[错误] '}${message}`,
            isStreaming: false,
          },
        ]
      }
      // done: just finalize streaming.
      return [...updated.slice(0, -1), { ...last, isStreaming: false }]
    })
    setIsSending(false)
    cleanupRef.current = null
  }, [rotateChatIdAfterForbidden])

  // BCDEF ticket 11: anomaly counters wired into every v2 stream
  // connection. The stream loop suppresses duplicate stream-terminals and
  // reports parser drops here, so both stay observable in state instead
  // of being silently accepted.
  const countDuplicateTerminal = useCallback(() => {
    v2StateRef.current = {
      ...v2StateRef.current,
      duplicateTerminalCount: v2StateRef.current.duplicateTerminalCount + 1,
    }
  }, [])
  const countDroppedFrame = useCallback(() => {
    v2StateRef.current = {
      ...v2StateRef.current,
      droppedCount: v2StateRef.current.droppedCount + 1,
    }
  }, [])

  // Ticket 11: attach a v2 SSE stream to a URL. Shared by the initial
  // /chat/sse request, the recovery effect's re-attach, and the resume
  // stream after a user decision. Centralizing the wiring avoids the
  // "attachV2Stream duplication" tech debt noted in Ticket 08 review.
  const attachV2Stream = useCallback((
    url: string,
    body: string,
    handlers: {
      onOpen?: () => void
      onEvent: (event: V2EventEnvelope) => void
      onTerminal: (event: V2EventEnvelope) => void
      onError: (error: string) => void
    },
  ) => {
    const cleanup = createV2StreamConnection(url, {
      body,
      onOpen: handlers.onOpen,
      onEvent: handlers.onEvent,
      onTerminal: handlers.onTerminal,
      onError: handlers.onError,
      // BCDEF ticket 11: anomalies are counted, never silently dropped.
      onDuplicateTerminal: countDuplicateTerminal,
      onDroppedFrame: countDroppedFrame,
    })
    cleanupRef.current = cleanup
  }, [countDuplicateTerminal, countDroppedFrame])

  // BCDEF ticket 11 (AC3): a live stream that closed WITHOUT a terminal is
  // NOT a connection error to fake and a run to discard — the run may be
  // perfectly healthy server-side. The snapshot + durable replay are the
  // authoritative recovery source: fetch /snapshot + /events?from_seq=
  // cursor, fold them into the reducer state, and finalize/continue the
  // UI from server truth. Never leaves the UI "streaming" when the
  // snapshot is terminal; never re-posts /chat/sse (no extra model call);
  // never creates a new conversation (canonical run_id/conversation_id
  // come from the persisted run + snapshot).
  const recoveredAssistantContent = useCallback((next: V2ConversationState): string => {
    let content = `${next.assistantText}${v2NoticeSuffix(next.reviewNotices)}`
    if (next.status === 'outcome_unknown') {
      content += `\n\n[结果未知] ${next.errorMessage || '操作结果未知，等待后台对账修复，不会自动重放。'}`
    } else if (next.status === 'error') {
      content += `\n\n[错误] ${next.errorMessage || '运行出错'}`
    } else if (next.status === 'cancelled') {
      content += `\n\n${USER_CANCELLED_MESSAGE}`
    }
    return content
  }, [])

  const reconcileAfterStreamLoss = useCallback(async () => {
    const persisted = v2RunRef.current
    if (!persisted) {
      setIsSending(false)
      cleanupRef.current = null
      return
    }
    const recoveryId = persisted.run_id || persisted.request_id

    const finalizeLocalWithoutServerTruth = (hint: string) => {
      setMessages(prev => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        if (last && last.role === 'assistant' && !last.content.includes(hint)) {
          const separator = last.content.trim() ? '\n\n' : ''
          updated[updated.length - 1] = {
            ...last,
            content: `${last.content}${separator}${hint}`,
            isStreaming: false,
          }
        } else if (last && last.role === 'assistant') {
          updated[updated.length - 1] = { ...last, isStreaming: false }
        }
        return updated
      })
      setIsSending(false)
      cleanupRef.current = null
    }

    let snapshot: V2RunSnapshot | null
    try {
      snapshot = await fetchV2Snapshot(recoveryId)
    } catch {
      // The snapshot endpoint itself is unreachable. Keep the persisted
      // run so the NEXT refresh retries the authoritative path — this is
      // deliberately not a faked [连接错误] and never clears the run.
      finalizeLocalWithoutServerTruth('[提示] 连接已中断，暂时无法获取运行快照；刷新页面可继续恢复该运行。')
      return
    }

    const applyRecovered = async (): Promise<'terminal' | 'paused' | 'running'> => {
      let replayEvents: V2EventEnvelope[] = []
      try {
        const replay = await fetchV2Events(snapshot?.run_id || recoveryId, persisted.sequence_cursor)
        replayEvents = replay.events.map(evt => ({
          version: '2',
          sequence: evt.sequence,
          type: evt.type,
          payload: evt.payload,
          request_id: evt.request_id,
          conversation_id: evt.conversation_id,
          run_id: evt.run_id,
          event_id: evt.event_id,
          timestamp: evt.timestamp,
        }))
      } catch {
        // Replay unavailable — the snapshot alone stays authoritative.
      }
      const next = reconcileV2Snapshot(reduceV2Events(v2StateRef.current, replayEvents), snapshot)
      // The snapshot messages are the fallback when replay cannot
      // complete the assistant turn (event row lost but message durable).
      if (!next.assistantText && snapshot) {
        const fallback = assistantTextFromSnapshot(snapshot, snapshot.run_id)
        if (fallback) next.assistantText = fallback
      }
      v2StateRef.current = next
      // Persisted cursor: never advance past a pause (AC6 — the paused
      // frame is only durable once the server confirms it, and replaying
      // the pause/approval pair is idempotent).
      if (v2RunRef.current) {
        v2RunRef.current = {
          ...v2RunRef.current,
          sequence_cursor:
            next.status === 'paused'
              ? persisted.sequence_cursor
              : Math.max(next.sequenceCursor, snapshot?.sequence_cursor ?? 0),
          pending_approval: next.pendingApproval,
          v2_status: next.status === 'idle' ? undefined : next.status,
        }
        saveV2PersistedRun(v2RunRef.current)
      }
      // Full rebuild (not append) — idempotent under repeated recovery:
      // the same state always renders the same single message (AC8).
      setMessages(prev => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        if (last && last.role === 'assistant') {
          updated[updated.length - 1] = {
            ...last,
            content: recoveredAssistantContent(next),
            isStreaming: !isV2Terminal(next) && !isV2Paused(next),
            artifacts: next.artifacts.length > 0 ? next.artifacts : last.artifacts,
          }
        }
        return updated
      })
      if (isV2Terminal(next)) {
        v2RunRef.current = null
        saveV2PersistedRun(null)
        setV2ApprovalState('idle')
        setPendingApproval(null)
        setIsSending(false)
        cleanupRef.current = null
        return 'terminal'
      }
      if (isV2Paused(next)) {
        setV2ApprovalState('idle')
        setPendingApproval(next.pendingApproval ?? persisted.pending_approval ?? null)
        cleanupRef.current = null
        return 'paused'
      }
      return 'running'
    }

    const outcome = await applyRecovered()
    if (outcome !== 'running') return

    if (snapshot === null) {
      // The server has no durable run at all — nothing to reconcile.
      // Finalize the local turn with a stable note (NOT a faked network
      // error, NOT a cleared run with lost content).
      v2RunRef.current = null
      saveV2PersistedRun(null)
      finalizeLocalWithoutServerTruth('[提示] 本次运行未能形成可恢复的服务记录，请重新发送。')
      return
    }

    // Snapshot still running: poll the authoritative source until the run
    // terminalizes / pauses (same bounded polling as mount recovery).
    const deadline = Date.now() + RECOVERY_POLL_TIMEOUT_MS
    const poll = async () => {
      if (!v2RunRef.current) return
      let next: V2RunSnapshot | null = null
      try {
        next = await fetchV2Snapshot(recoveryId)
      } catch {
        // transient — keep polling until the deadline.
      }
      if (!v2RunRef.current) return
      if (next === null) {
        v2RunRef.current = null
        saveV2PersistedRun(null)
        finalizeLocalWithoutServerTruth('[提示] 运行记录已不可查，请刷新页面。')
        return
      }
      snapshot = next
      if (V2_TERMINAL_RUN_STATUSES.has(next.status) || next.status === 'paused') {
        void applyRecovered()
        return
      }
      if (Date.now() >= deadline) {
        finalizeLocalWithoutServerTruth('[提示] 任务耗时较长，后台仍在执行；可稍后刷新查看结果。')
        return
      }
      setTimeout(() => { void poll() }, RECOVERY_POLL_INTERVAL_MS)
    }
    setTimeout(() => { void poll() }, RECOVERY_POLL_INTERVAL_MS)
  }, [recoveredAssistantContent])

  // Ticket 11: submit the user's approve/reject decision and open the
  // resume SSE stream. Spec AC4 + AC6: the decision is idempotent w.r.t.
  // decisionRequestId, and the resume stream reuses the same run_id +
  // sandbox + checkpoint. Failures (expired / lease lost / already
  // resolved) surface a user-facing message via v2ApprovalState='error'
  // and v2StateRef.current.resumeErrorMessage.
  const handleV2Approval = useCallback(async (
    decision: 'approve' | 'reject',
  ) => {
    const approval = v2StateRef.current.pendingApproval
    const persisted = v2RunRef.current
    if (!approval || !persisted || !persisted.run_id) {
      // Defensive: the UI should not render the buttons when these are
      // missing, but we guard against race conditions.
      return
    }
    if (v2ApprovalState === 'submitting' || v2ApprovalState === 'resuming') {
      // Spec AC5: duplicate clicks are no-ops. The server's idempotency
      // layer would catch them too, but we short-circuit to avoid an
      // extra network round-trip and a confusing UI flicker.
      return
    }

    setV2ApprovalState('submitting')
    const decisionRequestId = `dec-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`

    let decisionResponse: V2ApprovalDecisionResponse | null = null
    try {
      decisionResponse = await submitV2Approval(approval.approval_id, decision, decisionRequestId)
    } catch (error) {
      const message = error instanceof Error ? error.message : '审批决策失败'
      // Parse the structured 409 reason so the UI can render the right
      // message (e.g. "审批已过期" vs "沙箱租赁已丢失"). The format is
      // ``409:REASON:MESSAGE`` from submitV2Approval.
      const reasonMatch = /^409:([^:]+):(.+)$/.exec(message)
      const userMessage = reasonMatch ? reasonMatch[2] : message
      v2StateRef.current = {
        ...v2StateRef.current,
        resumeErrorMessage: userMessage,
      }
      setV2ApprovalState('error')
      // Persist the error so a refresh re-shows it (the run is still
      // paused server-side; the user can retry or cancel).
      if (v2RunRef.current) {
        v2RunRef.current = {
          ...v2RunRef.current,
          pending_approval: v2StateRef.current.pendingApproval,
          v2_status: 'paused',
        }
        saveV2PersistedRun(v2RunRef.current)
      }
      return
    }

    // Decision recorded successfully. Open the resume SSE stream from
    // the current sequence cursor so we don't replay already-seen
    // events. The server validates the run is no longer paused and the
    // sandbox lease is still active before emitting the continuation.
    setV2ApprovalState('resuming')
    // A10: the decision response carries the canonical resume URL — the
    // backend owns the gateway path. Fall back to the documented path
    // when the field is absent.
    //
    // AC6 stores ``sequence_cursor`` one frame back at ``run.paused`` so a
    // refresh can replay the pause prompt. Resuming must NOT reuse that
    // held-back cursor: the replay would re-deliver the already-consumed
    // ``run.paused`` and the stream client would treat it as terminal and
    // abort the resume before the continuation (the reducer cursor has
    // already advanced past the pause).
    const resumeFrom = Math.max(
      persisted.sequence_cursor,
      v2StateRef.current.sequenceCursor,
    )
    const resumeUrl = buildV2ResumeUrlFromDecision(
      decisionResponse ?? { resume_url: '', run_id: persisted.run_id },
      resumeFrom,
    )
    attachV2Stream(resumeUrl, '', {
      onOpen: () => setConnectionStatus('connected'),
      onEvent: handleV2Event,
      onTerminal: handleV2Terminal,
      onError: (error) => {
        // BCDEF ticket 11: a resume stream that closed without a terminal
        // reconciles against snapshot/replay too — the decision was already
        // recorded server-side, so this is never a "恢复失败" to fake and
        // never a re-decision or extra model call.
        if (error === 'stream_closed_without_terminal') {
          void reconcileAfterStreamLoss()
          return
        }
        setConnectionStatus('error')
        v2StateRef.current = {
          ...v2StateRef.current,
          resumeErrorMessage: error,
        }
        setV2ApprovalState('error')
        setMessages(prev => {
          const updated = [...prev]
          const last = updated[updated.length - 1]
          if (last && last.role === 'assistant') {
            updated[updated.length - 1] = {
              ...last,
              content: `${last.content}\n\n[恢复失败] ${error}`,
              isStreaming: false,
            }
          }
          return updated
        })
        setIsSending(false)
        cleanupRef.current = null
      },
    })
  }, [v2ApprovalState, attachV2Stream, handleV2Event, handleV2Terminal, reconcileAfterStreamLoss])

  // Ticket 08 + 11 + A06: refresh recovery as an explicit state machine
  // driven by planV2Recovery. On mount: a locally terminal run skips the
  // network entirely; otherwise one snapshot fetch decides between clear
  // (run unknown), single replay (terminal), approval re-show (paused),
  // or replay + snapshot polling (still running). Polling replaces the
  // old stream re-attach, which replayed without a terminal and surfaced
  // as a spurious stream error.
  useEffect(() => {
    const persisted = v2RunRef.current
    if (!persisted) return
    const persistedRun = persisted
    let cancelled = false
    let pollTimer: ReturnType<typeof setTimeout> | null = null

    // A05: snapshot/events resolve the request_id too, so the recovery
    // identifier is run_id || request_id — the empty run_id case (refresh
    // before the first event) recovers via the idempotency key.
    const recoveryId = persistedRun.run_id || persistedRun.request_id

    const clearPersisted = () => {
      v2RunRef.current = null
      saveV2PersistedRun(null)
      setV2ApprovalState('idle')
      setPendingApproval(null)
    }

    const applyReplay = async (snapshot: V2RunSnapshot | null): Promise<V2ConversationState> => {
      const replay = await fetchV2Events(recoveryId, persistedRun.sequence_cursor)
      const replayEnvelopes: V2EventEnvelope[] = replay.events.map(evt => ({
        version: '2',
        sequence: evt.sequence,
        type: evt.type,
        payload: evt.payload,
        request_id: evt.request_id,
        conversation_id: evt.conversation_id,
        run_id: evt.run_id,
        event_id: evt.event_id,
        timestamp: evt.timestamp,
      }))
      let next = reconcileV2Snapshot(
        reduceV2Events(v2StateRef.current, replayEnvelopes),
        snapshot,
      )
      // AC3: snapshot ``messages`` are the fallback when the replay
      // cannot complete the picture (assistant row durable in the
      // conversation DB but the SSE event row was lost).
      if (!next.assistantText && snapshot) {
        const fallback = assistantTextFromSnapshot(snapshot, snapshot.run_id)
        if (fallback) next = { ...next, assistantText: fallback }
      }
      v2StateRef.current = next

      // Tickets 05/06: rebuild the timeline from reducer state. A replay
      // that carries no trace-worthy events (legacy run, or a run whose
      // progress-class events were dropped) leaves whatever the stored
      // message already shows instead of wiping the user's history.
      const replayedTraceEvents = buildRecoveredTrace(next)

      setMessages(prev => {
        if (prev.length === 0) {
          // Fresh page load: restore the user message + assistant turn.
          const restored: Message[] = []
          if (persistedRun.user_message) {
            restored.push({ role: 'user', content: persistedRun.user_message })
          }
          const isActive = next.status === 'running' || next.status === 'paused'
          // BCDEF ticket 11: a terminal recovery (done/error/cancelled/
          // outcome_unknown) with no streamed text still needs the
          // assistant turn — that is where the stable terminal renders.
          if (next.assistantText || isActive || isV2Terminal(next)) {
            restored.push({
              role: 'assistant',
              // Review fix: replayed effect.outcome_unknown events live in
              // reviewNotices (the live path appends them to content
              // directly) — re-append so the replay preserves them. The
              // terminal suffix (error / 结果未知 / cancelled) renders the
              // one terminal distinctly and never as plain text (AC5/AC7).
              content: recoveredAssistantContent(next),
              isStreaming: !isV2Terminal(next) && !isV2Paused(next),
              // A02: artifacts delivered before the refresh are part of
              // the replay state — reattach so download cards survive.
              artifacts: next.artifacts.length > 0 ? next.artifacts : undefined,
              // Tickets 05/06: the inline timeline is rebuilt from the
              // reducer's ordered trace entries, so a refresh keeps the
              // tool calls, model rounds and reasoning already seen.
              traceEvents: replayedTraceEvents.length > 0 ? replayedTraceEvents : undefined,
            })
          }
          return restored
        }
        // Polling replays: refresh the trailing assistant message in
        // place so text/artifacts produced while polling show up when
        // the run finishes. Full rebuild from reducer state — repeated
        // recovery renders the SAME single message (AC8: no duplicated
        // UI text on repeated recovery).
        const updated = [...prev]
        const last = updated[updated.length - 1]
        if (last && last.role === 'assistant') {
          updated[updated.length - 1] = {
            ...last,
            content: recoveredAssistantContent(next),
            isStreaming: !isV2Terminal(next) && !isV2Paused(next),
            artifacts: next.artifacts.length > 0 ? next.artifacts : undefined,
            traceEvents:
              replayedTraceEvents.length > 0 ? replayedTraceEvents : last.traceEvents,
          }
        }
        return updated
      })

      // Update the persisted cursor so the next recovery starts from
      // here. AC6: a paused run keeps the ORIGINAL cursor — the pause is
      // not yet a durable server fact until the snapshot confirms the
      // continuation, and replaying the approval pair is idempotent.
      v2RunRef.current = {
        ...persistedRun,
        sequence_cursor:
          next.status === 'paused'
            ? persistedRun.sequence_cursor
            : Math.max(next.sequenceCursor, snapshot?.sequence_cursor ?? 0),
        pending_approval: next.pendingApproval ?? persistedRun.pending_approval,
        v2_status: next.status === 'idle' ? undefined : next.status,
      }
      saveV2PersistedRun(v2RunRef.current)
      return next
    }

    const showApprovalPrompt = (next: V2ConversationState) => {
      setV2ApprovalState('idle')
      // AC6: the approval survives refresh via three sources in order —
      // replayed ``approval.required``, snapshot ``pending_approvals``
      // (folded in by reconcileV2Snapshot), and the persisted run mirror.
      setPendingApproval(next.pendingApproval ?? persistedRun.pending_approval ?? null)
    }

    const startPolling = () => {
      const deadline = Date.now() + RECOVERY_POLL_TIMEOUT_MS
      const schedule = () => {
        if (Date.now() >= deadline) {
          // Cap: stop polling and hand the decision back to the user.
          setMessages(prev => {
            const updated = [...prev]
            const last = updated[updated.length - 1]
            if (last && last.role === 'assistant') {
              const separator = last.content.trim() ? '\n\n' : ''
              updated[updated.length - 1] = {
                ...last,
                content: `${last.content}${separator}[提示] 任务耗时较长，后台仍在执行；可稍后刷新查看结果。`,
                isStreaming: false,
              }
            }
            return updated
          })
          return
        }
        pollTimer = setTimeout(() => { void poll() }, RECOVERY_POLL_INTERVAL_MS)
      }
      const poll = async () => {
        if (cancelled) return
        try {
          const snapshot = await fetchV2Snapshot(recoveryId)
          if (cancelled) return
          if (snapshot === null) {
            // Run row vanished mid-poll (server reaped it) — render
            // whatever the event log still holds and stop.
            await applyReplay(null)
            if (cancelled) return
            clearPersisted()
            return
          }
          if (V2_TERMINAL_RUN_STATUSES.has(snapshot.status)) {
            await applyReplay(snapshot)
            if (cancelled) return
            clearPersisted()
            return
          }
          if (snapshot.status === 'paused') {
            // Approval became necessary while polling — re-show the
            // prompt; the persisted run stays for handleV2Approval.
            const next = await applyReplay(snapshot)
            if (cancelled) return
            showApprovalPrompt(next)
            return
          }
          schedule()
        } catch {
          // Transient network error — keep polling until the deadline.
          schedule()
        }
      }
      schedule()
    }

    async function recoverV2Run() {
      try {
        // First pass without a snapshot: only the local-only decisions
        // (skip / local-terminal) can be made here.
        const prePlan = planV2Recovery(persistedRun, null)
        if (prePlan.kind === 'skip' || prePlan.kind === 'local-terminal') return

        const snapshot = await fetchV2Snapshot(recoveryId)
        if (cancelled) return
        const plan = planV2Recovery(persistedRun, snapshot ? snapshot.status : null)

        if (plan.kind === 'clear') {
          // Run unknown on the server (e.g. DB reset). Clear local state.
          clearPersisted()
          return
        }
        if (plan.kind === 'replay-terminal') {
          await applyReplay(snapshot)
          if (cancelled) return
          clearPersisted()
          return
        }
        const next = await applyReplay(snapshot)
        if (cancelled) return
        if (isV2Terminal(next)) {
          clearPersisted()
          return
        }
        if (isV2Paused(next)) {
          // Ticket 11 spec AC6: paused run — do NOT re-attach a stream.
          // The approve/reject UI renders from next.pendingApproval.
          showApprovalPrompt(next)
          return
        }
        startPolling()
      } catch {
        // Recovery failed (network/snapshot error). Leave the persisted
        // run in place; the user can retry by refreshing again.
      }
    }

    void recoverV2Run()
    return () => {
      cancelled = true
      if (pollTimer) clearTimeout(pollTimer)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Persist
  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, serializeMessages(messages)) } catch { /* ignore */ }
  }, [messages])

  // Persist sessions
  useEffect(() => {
    saveSessions(sessions)
  }, [sessions])

  useEffect(() => {
    if (activeSessionId) {
      try { localStorage.setItem(ACTIVE_SESSION_KEY, activeSessionId) } catch { /* ignore */ }
    } else {
      try { localStorage.removeItem(ACTIVE_SESSION_KEY) } catch { /* ignore */ }
    }
  }, [activeSessionId])

  useEffect(() => {
    if (!activeSessionId || messages.length === 0) return
    const lastMsg = messages[messages.length - 1]
    if (lastMsg?.isStreaming) return
    const firstUserMsg = messages.find(m => m.role === 'user')
    const title = firstUserMsg ? firstUserMsg.content.slice(0, 30) + (firstUserMsg.content.length > 30 ? '...' : '') : '新对话'
    setSessions(prev => {
      const existing = prev.find(s => s.id === activeSessionId)
      if (existing) {
        return prev.map(s => s.id === activeSessionId ? { ...s, chatId: chatIdRef.current, title, date: getTodayDate(), messages } : s)
      }
      return [{ id: activeSessionId, chatId: chatIdRef.current, title, date: getTodayDate(), messages }, ...prev].slice(0, 50)
    })
  }, [activeSessionId, messages])

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => { scrollToBottom() }, [messages, scrollToBottom])

  useEffect(() => { return () => { cleanupRef.current?.() } }, [])

  // 连接状态由 v2 SSE onOpen / onError 驱动（v1 health 轮询已随 v1 移除）。

  const appendCancellationNotice = useCallback(() => {
    setMessages(prev => {
      const updated = [...prev]
      const last = updated[updated.length - 1]
      if (last && last.role === 'assistant') {
        if (last.content.includes(USER_CANCELLED_MESSAGE)) {
          updated[updated.length - 1] = { ...last, isStreaming: false }
          return updated
        }
        const separator = last.content.trim() ? '\n\n' : ''
        updated[updated.length - 1] = {
          ...last,
          content: `${last.content}${separator}${USER_CANCELLED_MESSAGE}`,
          isStreaming: false,
        }
      }
      return updated
    })
  }, [])

  // 会话轮换（新建/切换/清空/删除当前会话）时拆除仍在运行的流。
  // 不拆的话，旧 run 的 v2 事件回调仍会向已切换的 messages 追加
  // 内容，串进新会话。v2 run 尽力发出幂等取消。
  const teardownActiveStream = useCallback(() => {
    if (cleanupRef.current) {
      cleanupRef.current()
      cleanupRef.current = null
    }
    const persisted = v2RunRef.current
    if (persisted) {
      // 取消端点幂等且接受 request_id 兜底（后端 spec AC4）；失败不
      // 影响本地收尾，服务端 run 交给 janitor/超时收敛。
      void cancelV2Run(
        persisted.run_id || persisted.request_id,
        `teardown-${Date.now()}`
      ).catch(() => undefined)
      v2RunRef.current = null
      saveV2PersistedRun(null)
      v2StateRef.current = { ...v2StateRef.current, status: 'cancelled' }
      setV2ApprovalState('idle')
      setPendingApproval(null)
    }
    setIsSending(false)
  }, [])

  const createNewSession = useCallback(() => {
    teardownActiveStream()
    if (messages.length > 0 && activeSessionId) {
      const firstUserMsg = messages.find(m => m.role === 'user')
      const title = firstUserMsg ? firstUserMsg.content.slice(0, 30) + (firstUserMsg.content.length > 30 ? '...' : '') : '新对话'
      setSessions(prev => {
        const existing = prev.find(s => s.id === activeSessionId)
        if (existing) {
          return prev.map(s => s.id === activeSessionId ? { ...s, chatId: chatIdRef.current, title, date: getTodayDate(), messages } : s)
        }
        return [{ id: activeSessionId, chatId: chatIdRef.current, title, date: getTodayDate(), messages }, ...prev].slice(0, 50)
      })
    }
    const newId = generateId()
    const next = createChatId()
    setActiveSessionId(newId)
    setMessages([])
    localStorage.setItem(CHAT_ID_KEY, next)
    chatIdRef.current = next
    try { localStorage.removeItem(STORAGE_KEY) } catch { /* ignore */ }
  }, [messages, activeSessionId, teardownActiveStream])

  const loadSession = useCallback((session: ChatSession) => {
    teardownActiveStream()
    const target = session.chatId ? session : { ...session, chatId: createChatId() }
    if (activeSessionId && messages.length > 0) {
      const firstUserMsg = messages.find(m => m.role === 'user')
      const title = firstUserMsg ? firstUserMsg.content.slice(0, 30) + (firstUserMsg.content.length > 30 ? '...' : '') : '新对话'
      setSessions(prev => {
        const existing = prev.find(s => s.id === activeSessionId)
        if (existing) {
          return prev.map(s => s.id === activeSessionId ? { ...s, chatId: chatIdRef.current, title, date: getTodayDate(), messages } : s)
        }
        return [{ id: activeSessionId, chatId: chatIdRef.current, title, date: getTodayDate(), messages }, ...prev].slice(0, 50)
      })
    }
    setActiveSessionId(target.id)
    chatIdRef.current = target.chatId
    localStorage.setItem(CHAT_ID_KEY, target.chatId)
    setMessages(target.messages)
    try { localStorage.setItem(STORAGE_KEY, serializeMessages(target.messages)) } catch { /* ignore */ }
  }, [activeSessionId, messages, teardownActiveStream])

  const selectSession = useCallback((sessionId: string) => {
    const target = sessions.find(s => s.id === sessionId)
    if (target) loadSession(target)
  }, [sessions, loadSession])

  const deleteSession = useCallback((sessionId: string) => {
    setSessions(prev => prev.filter(s => s.id !== sessionId))
    if (activeSessionId === sessionId) {
      // 删除的是当前活跃会话：先拆除活动流,避免旧 run 的事件写入轮换后的新会话。
      teardownActiveStream()
      setActiveSessionId(null)
      setMessages([])
      const next = createChatId()
      localStorage.setItem(CHAT_ID_KEY, next)
      chatIdRef.current = next
      try { localStorage.removeItem(STORAGE_KEY) } catch { /* ignore */ }
    }
  }, [activeSessionId, teardownActiveStream])

  const handleSend = useCallback(async (
    textOverride?: string,
    options: { replaceFromIndex?: number } = {},
  ) => {
    const text = typeof textOverride === 'string' ? textOverride.trim() : input.trim()
    if (!text || isSending) return

    if (!activeSessionId) {
      setActiveSessionId(generateId())
    }

    setInput('')
    setSelectedScenario(null)
    setIsSending(true)

    const now = new Date()
    const timeStr = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`
    const chatId = chatIdRef.current
    const replaceFromIndex = options.replaceFromIndex
    const contextMessages = typeof replaceFromIndex === 'number'
      ? messages.slice(0, Math.max(0, replaceFromIndex))
      : messages
    const nextUserMessage: Message = { role: 'user', content: text, time: timeStr }
    const nextAssistantMessage: Message = { role: 'assistant', content: '', isStreaming: true }

    setMessages(prev => {
      const base = typeof replaceFromIndex === 'number'
        ? prev.slice(0, Math.max(0, replaceFromIndex))
        : prev
      return [...base, nextUserMessage, nextAssistantMessage]
    })

    // Ticket 08: v2 是唯一 chat 链路。Spec AC1 requires an independent v2
    // client and v2 event reducer. v2 does NOT use a typewriter queue —
    // text deltas are written directly to the messages array. The v2
    // request body is strict: only message, request_id, conversation_id,
    // profile_id, and current_location are allowed (server rejects
    // everything else).
    {
      const requestId = `req-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
      const conversationId = chatId
      // Reset v2 per-run state. The reducer's sequence monotonicity guard
      // makes this safe even if a stale event from a previous run slips in.
      v2StateRef.current = initialV2State
      v2RunRef.current = {
        run_id: '', // filled in by the server; we use request_id for reconnect
        request_id: requestId,
        conversation_id: conversationId,
        sequence_cursor: 0,
        user_message: text,
      }
      saveV2PersistedRun(v2RunRef.current)

      try {
        const v2Payload: Record<string, unknown> = {
          message: text,
          request_id: requestId,
          conversation_id: conversationId,
          profile_id: selectedScenario === 'route_plan' ? 'route_plan' : 'chat',
        }
        const needsRouteFollowupLocation = shouldRequestRouteFollowupLocation(text, contextMessages)
        const cachedRouteLocation = needsRouteFollowupLocation ? lastLocationRef.current : null
        const currentLocation = cachedRouteLocation
          || await getBrowserLocation((shouldRequestCurrentLocation(text) || needsRouteFollowupLocation) ? 10000 : 3000)
        if (currentLocation) {
          lastLocationRef.current = currentLocation
          v2Payload.current_location = currentLocation
        } else if (needsRouteFollowupLocation && lastLocationRef.current) {
          v2Payload.current_location = lastLocationRef.current
        }

        const cleanup = createV2StreamConnection(API.chatV2SSE, {
          body: JSON.stringify(v2Payload),
          onOpen: () => setConnectionStatus('connected'),
          onEvent: handleV2Event,
          onTerminal: handleV2Terminal,
          onDuplicateTerminal: countDuplicateTerminal,
          onDroppedFrame: countDroppedFrame,
          onError: (error) => {
            // BCDEF ticket 11: transport closure without a terminal is a
            // RECOVERY trigger, not a connection error. Reconcile against
            // the authoritative snapshot + replay instead of appending a
            // fake [连接错误] and clearing the persisted run.
            if (error === 'stream_closed_without_terminal') {
              void reconcileAfterStreamLoss()
              return
            }
            setConnectionStatus('error')
            setMessages(prev => {
              const updated = [...prev]
              const last = updated[updated.length - 1]
              if (last && last.role === 'assistant') {
                updated[updated.length - 1] = {
                  ...last,
                  content: `${last.content}\n\n[连接错误] ${error}`,
                  isStreaming: false,
                }
              }
              return updated
            })
            v2RunRef.current = null
            saveV2PersistedRun(null)
            setIsSending(false)
            cleanupRef.current = null
          },
        })
        cleanupRef.current = cleanup
      } catch (error) {
        setMessages(prev => {
          const updated = [...prev]
          const last = updated[updated.length - 1]
          if (last && last.role === 'assistant') {
            updated[updated.length - 1] = {
              ...last,
              content: `${last.content}\n\n[错误] ${error instanceof Error ? error.message : '请求失败'}`,
              isStreaming: false,
            }
          }
          return updated
        })
        v2RunRef.current = null
        saveV2PersistedRun(null)
        setIsSending(false)
      }
    }
  }, [input, isSending, activeSessionId, selectedScenario, messages, handleV2Event, handleV2Terminal, reconcileAfterStreamLoss, countDuplicateTerminal, countDroppedFrame])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleClear = () => {
    teardownActiveStream()
    setMessages([])
    try {
      localStorage.removeItem(STORAGE_KEY)
      const next = createChatId()
      localStorage.setItem(CHAT_ID_KEY, next)
      chatIdRef.current = next
    } catch { /* ignore */ }
  }

  /** Regenerate: remove the last assistant response, re-send the last user prompt */
  const handleRegenerate = useCallback(() => {
    if (isSending) return
    // Find the last user message (reverse search for TS < es2023 compat)
    let lastUserIdx = -1
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') { lastUserIdx = i; break }
    }
    if (lastUserIdx === -1) return
    const lastUserText = messages[lastUserIdx].content
    handleSend(lastUserText, { replaceFromIndex: lastUserIdx })
  }, [messages, isSending, handleSend])

  const runtimeAdapter: ExternalStoreAdapter<ThreadMessage> = {
    messages: messages.map(convertMessage),
    isRunning: isSending || messages.some(m => m.isStreaming),
    onNew: async (appendMessage) => {
      const text = appendMessage.content.find((p): p is {type: "text", text: string} => p.type === "text")?.text || "";
      handleSend(text);
    },
    onCancel: async () => {
      if (cleanupRef.current) {
        cleanupRef.current();
        cleanupRef.current = null;
      }
      // Ticket 08: v2 cancel (spec AC4). When a v2 run is active, call the
      // idempotent cancel endpoint. The server returns accepted=true (run
      // transitioned to cancelled) or accepted=false (already terminal).
      // Either way, the streaming side has already been torn down by
      // cleanupRef. The terminal `cancelled` event (if it arrives via
      // reconnect) appends the user-visible notice; if it doesn't arrive
      // (e.g. server already gone), we append the notice here so the UI
      // never stays in a "thinking" state.
      if (v2RunRef.current) {
        const persisted = v2RunRef.current
        let serverConfirmed = false
        try {
          await cancelV2Run(persisted.run_id || persisted.request_id, `cancel-${Date.now()}`)
          serverConfirmed = true
        } catch {
          // Best-effort: if the cancel API is unreachable we still
          // finalize the UI locally — but BCDEF ticket 11 no longer
          // discards the recovery context on an UNCONFIRMED cancel. The
          // persisted run stays (canonical run/request id + cursor), so
          // the next refresh reconciles against snapshot/replay — the
          // authoritative source after cancel — instead of leaving a
          // server-side run nobody can find.
        }
        // Update v2 state to cancelled locally so the UI shows the right
        // status even if the terminal event never arrives.
        v2StateRef.current = { ...v2StateRef.current, status: 'cancelled' }
        if (serverConfirmed) {
          // The server holds the cancelled terminal and it replays (AC4:
          // cancelled is a replayable terminal) — safe to clear locally.
          v2RunRef.current = null
          saveV2PersistedRun(null)
        }
        setPendingApproval(null)
        appendCancellationNotice()
      }
      setIsSending(false)
    }
  };

  return (
    <SessionRuntimeProvider key={chatIdRef.current} adapter={runtimeAdapter}>
      <div className="flex h-full w-full overflow-hidden relative">
        {/* Gemini 风格停靠侧栏：桌面折叠栏 + 移动端抽屉 */}
        <GeminiSidebar
          collapsed={sidebarCollapsed}
          onCollapsedChange={setSidebarCollapsed}
          mobileOpen={mobileSidebarOpen}
          onMobileOpenChange={setMobileSidebarOpen}
          sessions={sessions}
          activeSessionId={activeSessionId}
          onNewSession={createNewSession}
          onSelectSession={selectSession}
          onDeleteSession={deleteSession}
        />

        {/* Main Gemini Thread Area */}
        <div className="min-w-0 flex-1 flex flex-col relative overflow-hidden bg-transparent">
          {v2ApprovalState === 'error' && v2StateRef.current.resumeErrorMessage && (
            <div className="absolute left-4 right-4 top-16 z-50 ml-auto max-w-sm rounded-md border border-red-400/30 bg-zinc-950 p-4 text-sm text-white shadow-xl">
              <div className="font-medium text-red-300">恢复失败</div>
              <div className="mt-2 break-words text-white/75">{v2StateRef.current.resumeErrorMessage}</div>
            </div>
          )}
          {pendingApproval && v2ApprovalState !== 'resolved' && (
            <div className="absolute left-4 right-4 top-16 z-50 ml-auto max-w-sm rounded-md border border-amber-400/30 bg-zinc-950 p-4 text-sm text-white shadow-xl">
              <div className="font-medium">
                工具审批：{pendingApproval.tool}
                <span className="ml-2 text-xs text-amber-300">风险 {pendingApproval.risk_level}</span>
              </div>
              {pendingApproval.message && <div className="mt-1 text-white/70">{pendingApproval.message}</div>}
              <div className="mt-2 break-words text-white/75">{pendingApproval.impact}</div>
              <div className="mt-3 flex justify-end gap-2">
                <button
                  className="px-3 py-1.5 text-white/70 hover:text-white disabled:opacity-50"
                  disabled={v2ApprovalState === 'submitting' || v2ApprovalState === 'resuming'}
                  onClick={() => { void handleV2Approval('reject') }}
                >
                  拒绝
                </button>
                <button
                  className="rounded-md bg-emerald-600 px-3 py-1.5 text-white hover:bg-emerald-500 disabled:opacity-50"
                  disabled={v2ApprovalState === 'submitting' || v2ApprovalState === 'resuming'}
                  onClick={() => { void handleV2Approval('approve') }}
                >
                  {v2ApprovalState === 'submitting' || v2ApprovalState === 'resuming' ? '处理中...' : '批准'}
                </button>
              </div>
            </div>
          )}
          <div className="absolute top-4 right-4 z-40">
            <DropdownMenu
              icon={<Settings2 className="h-4 w-4" />}
              items={[
                {
                  label: '模型配置',
                  icon: <SlidersHorizontal className="h-4 w-4" />,
                  onSelect: () => {
                    suppressMenuRefocus.current = true
                    setConfigOpen(true)
                  },
                },
                {
                  label: '长期记忆',
                  icon: <Database className="h-4 w-4" />,
                  separatorBefore: true,
                  onSelect: () => {
                    suppressMenuRefocus.current = true
                    setMemoryDialogOpen(true)
                  },
                },
                {
                  label: '知识源',
                  icon: <Database className="h-4 w-4" />,
                  separatorBefore: true,
                  onSelect: () => {
                    suppressMenuRefocus.current = true
                    setKnowledgeSourcesOpen(true)
                  },
                },
                {
                  label: 'MCP 接入',
                  icon: <Plug className="h-4 w-4" />,
                  separatorBefore: true,
                  onSelect: () => {
                    suppressMenuRefocus.current = true
                    setMcpDialogOpen(true)
                  },
                },
              ]}
              label="设置"
              triggerRef={settingsTriggerRef}
              onCloseAutoFocus={(event) => {
                if (!suppressMenuRefocus.current) return
                suppressMenuRefocus.current = false
                event.preventDefault()
              }}
            />
          </div>
          {configOpen && <LlmConfigDialog onClose={closeLlmConfig} />}
          {memoryDialogOpen && <MemoryDialog onClose={closeMemoryDialog} refreshKey={memoryRefreshKey} />}
          {mcpDialogOpen && <McpDialog onClose={closeMcpDialog} />}
          {knowledgeSourcesOpen && <KnowledgeSourcesDialog onClose={() => setKnowledgeSourcesOpen(false)} />}
          {pendingMemoryConsent && (
            <div className="absolute left-4 right-4 top-16 z-50 ml-auto max-w-sm rounded-md border border-amber-400/30 bg-zinc-950 p-4 text-sm text-white shadow-xl">
              <div className="font-medium">确认保存 {pendingMemoryConsent.category}</div>
              <div className="mt-2 break-words text-white/75">{pendingMemoryConsent.preview}</div>
              {memoryDecisionError && <div className="mt-2 text-xs text-red-300">{memoryDecisionError}</div>}
              <div className="mt-3 flex justify-end gap-2">
                <button className="px-3 py-1.5 text-white/70 hover:text-white" disabled={memoryDecisionPending} onClick={() => { void handleMemoryDecision('reject') }}>拒绝</button>
                <button className="rounded-md bg-emerald-600 px-3 py-1.5 text-white hover:bg-emerald-500 disabled:opacity-50" disabled={memoryDecisionPending} onClick={() => { void handleMemoryDecision('approve') }}>保存</button>
              </div>
            </div>
          )}
          <GeminiThread
            emptyTitle="告诉我你的徒步计划，我将为你规划路线、安排任务，并提供安全建议。"
            mapFocus={mapFocus}
            mapSessionKey={activeSessionId ?? 'new'}
            onRegenerate={handleRegenerate}
            realMessageCount={messages.length}
          />
        </div>
      </div>
    </SessionRuntimeProvider>
  )
}

export default SuperAgent
