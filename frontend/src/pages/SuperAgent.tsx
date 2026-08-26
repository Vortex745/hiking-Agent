import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import { API } from '../api/config'
import { buildRuntimeModelSettings } from '../api/llmConfig'
import { createStreamConnection, SSEEvent } from '../api/sse'
import { createTypewriterStreamQueue } from '../api/typewriterStream'
import {
  createV2StreamConnection,
  fetchChatRuntime,
  cancelV2Run,
  fetchV2Snapshot,
  fetchV2Events,
  submitV2Approval,
  buildV2ResumeStreamUrl,
  fetchV2MemorySettings,
  updateV2MemorySettings,
  submitV2MemoryDecision,
  fetchV2Memories,
  updateV2Memory,
  deleteV2Memory,
  clearV2Memories,
  exportV2Memories,
  type V2EventEnvelope,
  type V2RunSnapshot,
  type V2ApprovalRequest,
  type V2Memory,
  type V2MemoryCandidate,
  type ChatRuntime,
} from '../api/chatV2'
import {
  reduceV2Event,
  reduceV2Events,
  initialV2State,
  isV2Terminal,
  isV2Paused,
  type V2ConversationState,
} from '../api/v2Reducer'
import { useExternalStoreRuntime, AssistantRuntimeProvider, ThreadMessage } from "@assistant-ui/react"
import { GeminiThread } from '../components/assistant-ui/gemini/GeminiThread'
import { FloatingSidebar } from '../components/FloatingSidebar'
import { Menu, X, MessageSquare, Trash2, ClipboardList, CheckCircle2, AlertTriangle, Map, Database, Download, Pencil } from 'lucide-react'

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
  runtime?: ChatRuntime
}

type AgentScenario = 'route_plan' | 'gear_check' | 'risk_assessment' | 'report_export'
type AgentTraceEvent = Pick<SSEEvent, 'type' | 'content' | 'metadata'>
type AgentArtifact = Pick<SSEEvent, 'content' | 'metadata'>
type BrowserLocation = {
  latitude: number
  longitude: number
  accuracy?: number
  source: 'browser'
}
type ToolConfirmationAction = 'confirm' | 'reject'
type ToolConfirmationResponse = {
  status?: string
  confirmation_id?: string
  tool_name?: string
  tool_result?: Record<string, unknown>
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
  { icon: ClipboardList, text: '生成行程', scenario: 'route_plan' as AgentScenario },
  { icon: CheckCircle2, text: '检查装备', scenario: 'gear_check' as AgentScenario },
  { icon: AlertTriangle, text: '风险提醒', scenario: 'risk_assessment' as AgentScenario },
  { icon: Map, text: '推荐路线', scenario: 'route_plan' as AgentScenario },
]

const traceLabels: Partial<Record<SSEEvent['type'], string>> = {
  thought: '思考',
  tool_call: '工具调用',
  tool_result: '工具结果',
  approval_required: '需要确认',
}

const STORAGE_KEY = 'ai-hiking-agent-chat'
const SESSIONS_KEY = 'ai-hiking-agent-sessions'
const CHAT_ID_KEY = 'ai-hiking-agent-chat-id'
const ACTIVE_SESSION_KEY = 'ai-hiking-agent-active-session'
const SENSITIVE_LOCATION_FIELDS = new Set(['latitude', 'longitude', 'accuracy'])
const USER_CANCELLED_MESSAGE = '用户主动取消回答，请补充内容或点击重试。'

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
  v2_status?: 'running' | 'paused' | 'done' | 'error' | 'cancelled'
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

function normalizeServerMessages(payload: unknown): Message[] {
  const messages = (payload as { messages?: unknown })?.messages
  if (!Array.isArray(messages)) return []
  return messages
    .map(item => {
      const role = (item as { role?: unknown }).role
      const content = (item as { content?: unknown }).content
      if ((role !== 'user' && role !== 'assistant') || typeof content !== 'string') {
        return null
      }
      return { role, content } as Message
    })
    .filter((item): item is Message => item !== null)
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

function formatConfirmedToolResult(toolResult: Record<string, unknown>) {
  if (toolResult.ok === false) {
    return typeof toolResult.error === 'string' && toolResult.error
      ? `工具执行失败: ${toolResult.error}`
      : '工具执行失败'
  }
  const result = toolResult.result
  if (typeof result === 'string') return result
  if (result === null || result === undefined) return '工具已执行'
  return JSON.stringify(result)
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
      runtime: session.runtime === 'v2' ? 'v2' : 'v1',
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

  return {
    id: `msg-${index}`,
    role: msg.role,
    content: content,
    status: (msg.isStreaming ? { type: "running" } : { type: "complete" }),
    createdAt: new Date(),
    metadata: {},
  } as unknown as ThreadMessage;
};

function SuperAgent() {
  const [messages, setMessages] = useState<Message[]>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY)
      return saved ? JSON.parse(saved) : []
    } catch { return [] }
  })
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
  const [sidebarOpen, setSidebarOpen] = useState(() => typeof window !== 'undefined' ? window.innerWidth >= 768 : true)

  useEffect(() => {
    const handleToggle = () => setSidebarOpen(prev => !prev)
    window.addEventListener('toggle-history', handleToggle)
    return () => window.removeEventListener('toggle-history', handleToggle)
  }, [])

  const [input, setInput] = useState('')
  const [selectedScenario, setSelectedScenario] = useState<AgentScenario | null>(null)
  const [isSending, setIsSending] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connected' | 'error'>('disconnected')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const cleanupRef = useRef<(() => void) | null>(null)
  const chatIdRef = useRef(getOrCreateChatId())
  const lastLocationRef = useRef<BrowserLocation | null>(null)

  // Ticket 08: v2 mode + run state. v2Mode defaults off (spec AC7) and is
  // persisted so an internal tester's choice survives refresh. The v2 run
  // ref tracks the active run_id + sequence cursor so refresh recovery
  // can call /snapshot + /events to resume without re-sending the user
  // message or re-displaying already-shown text (spec AC3).
  //
  // Ticket 11: ``v2ApprovalState`` tracks the in-flight decision so the
  // UI can render "处理中" while the POST /approvals/{id}/decision
  // request is in flight, and "已解决" / "恢复失败" once the resume
  // stream opens or fails. It is reset to 'idle' on the next
  // ``approval.required`` event.
  const [v2Mode, setV2Mode] = useState<boolean>(() => loadV2PersistedRun() !== null)
  const [memoryEnabled, setMemoryEnabled] = useState(false)
  const [memoryPending, setMemoryPending] = useState(false)
  const [memoryPanelOpen, setMemoryPanelOpen] = useState(false)
  const [memoryItems, setMemoryItems] = useState<V2Memory[]>([])
  const [memoryTotal, setMemoryTotal] = useState(0)
  const [memoryOffset, setMemoryOffset] = useState(0)
  const [memoryListPending, setMemoryListPending] = useState(false)
  const [memoryError, setMemoryError] = useState('')
  const [pendingMemoryConsent, setPendingMemoryConsent] = useState<V2MemoryCandidate | null>(null)
  const [memoryDecisionPending, setMemoryDecisionPending] = useState(false)
  const v2RunRef = useRef<V2PersistedRun | null>(loadV2PersistedRun())
  const v2StateRef = useRef<V2ConversationState>(initialV2State)
  const [v2ApprovalState, setV2ApprovalState] = useState<
    'idle' | 'submitting' | 'resuming' | 'resolved' | 'error'
  >('idle')

  useEffect(() => {
    if (!v2Mode) {
      setMemoryEnabled(false)
      setMemoryPanelOpen(false)
      setPendingMemoryConsent(null)
      return
    }
    let cancelled = false
    void fetchV2MemorySettings()
      .then(settings => { if (!cancelled) setMemoryEnabled(settings.enabled) })
      .catch(() => { if (!cancelled) setMemoryEnabled(false) })
    return () => { cancelled = true }
  }, [v2Mode])

  const handleMemoryToggle = useCallback(async () => {
    if (memoryPending) return
    setMemoryPending(true)
    try {
      const settings = await updateV2MemorySettings(!memoryEnabled)
      setMemoryEnabled(settings.enabled)
    } finally {
      setMemoryPending(false)
    }
  }, [memoryEnabled, memoryPending])

  useEffect(() => {
    if (!v2Mode || !memoryPanelOpen) return
    let cancelled = false
    setMemoryListPending(true)
    setMemoryError('')
    void fetchV2Memories(20, memoryOffset)
      .then(page => {
        if (cancelled) return
        setMemoryItems(page.items)
        setMemoryTotal(page.total)
      })
      .catch(error => {
        if (!cancelled) setMemoryError(error instanceof Error ? error.message : '记忆加载失败')
      })
      .finally(() => { if (!cancelled) setMemoryListPending(false) })
    return () => { cancelled = true }
  }, [v2Mode, memoryPanelOpen, memoryOffset])

  const reloadMemories = useCallback(async () => {
    const page = await fetchV2Memories(20, memoryOffset)
    setMemoryItems(page.items)
    setMemoryTotal(page.total)
  }, [memoryOffset])

  const handleMemoryDecision = useCallback(async (decision: 'approve' | 'reject') => {
    if (!pendingMemoryConsent || memoryDecisionPending) return
    setMemoryDecisionPending(true)
    setMemoryError('')
    try {
      await submitV2MemoryDecision(
        pendingMemoryConsent.candidate_id,
        decision,
        `memory-decision-${Date.now()}`,
      )
      setPendingMemoryConsent(null)
      v2StateRef.current = { ...v2StateRef.current, pendingMemoryConsent: null }
      if (memoryPanelOpen) await reloadMemories()
    } catch (error) {
      setMemoryError(error instanceof Error ? error.message : '记忆决策失败')
    } finally {
      setMemoryDecisionPending(false)
    }
  }, [memoryDecisionPending, memoryPanelOpen, pendingMemoryConsent, reloadMemories])

  const handleMemoryEdit = useCallback(async (item: V2Memory) => {
    const content = window.prompt('记忆内容', item.content)
    if (content === null || !content.trim()) return
    const category = window.prompt('类别', item.category)
    if (category === null || !category.trim()) return
    const ttlInput = window.prompt('有效期天数（留空表示永久）', '')
    if (ttlInput === null) return
    const ttlDays = ttlInput.trim() ? Number(ttlInput) : null
    if (ttlDays !== null && (!Number.isInteger(ttlDays) || ttlDays < 1 || ttlDays > 3650)) {
      setMemoryError('有效期必须是 1 到 3650 天的整数')
      return
    }
    try {
      await updateV2Memory(item.memory_id, {
        content: content.trim(),
        category: category.trim(),
        ttl_days: ttlDays,
      })
      await reloadMemories()
    } catch (error) {
      setMemoryError(error instanceof Error ? error.message : '记忆更新失败')
    }
  }, [reloadMemories])

  const handleMemoryDelete = useCallback(async (memoryId: string) => {
    try {
      await deleteV2Memory(memoryId)
      await reloadMemories()
    } catch (error) {
      setMemoryError(error instanceof Error ? error.message : '记忆删除失败')
    }
  }, [reloadMemories])

  const handleMemoryClear = useCallback(async () => {
    if (!window.confirm('确认清除全部长期记忆？此操作无法撤销。')) return
    try {
      await clearV2Memories()
      setMemoryOffset(0)
      await reloadMemories()
    } catch (error) {
      setMemoryError(error instanceof Error ? error.message : '记忆清除失败')
    }
  }, [reloadMemories])

  const handleMemoryExport = useCallback(async () => {
    try {
      const exported = await exportV2Memories()
      window.open(API.chatV2Artifact(exported.artifact_id), '_blank', 'noopener,noreferrer')
    } catch (error) {
      setMemoryError(error instanceof Error ? error.message : '记忆导出失败')
    }
  }, [])

  // Ticket 08: v2 event handlers (spec AC2). Map v2 wire events to stable
  // UI state. These are defined as useCallbacks before the recovery effect
  // so the effect can reference them without forward-reference issues.
  // v2 text is written directly to the messages array (no typewriter queue)
  // to keep v2 independent of v1's streaming polish (spec AC1).
  const handleV2Event = useCallback((event: V2EventEnvelope) => {
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
    if (v2RunRef.current) {
      v2RunRef.current = {
        ...v2RunRef.current,
        sequence_cursor: next.sequenceCursor,
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
    }
    // Ticket 11: ``approval.resolved`` arrives on the resume stream —
    // the decision has been recorded server-side and the run is
    // continuing. Mark the UI as resolved so the approve/reject buttons
    // dismiss gracefully while the next events arrive.
    if (event.type === 'approval.resolved') {
      setV2ApprovalState('resolved')
    }
    if (event.type === 'memory.consent_required') {
      setPendingMemoryConsent(next.pendingMemoryConsent)
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
    } else if (event.type === 'tool_call' || event.type === 'tool_result') {
      const traceEvent: AgentTraceEvent = {
        type: event.type,
        content: typeof event.payload.content === 'string' ? event.payload.content : '',
        metadata: event.payload as Record<string, unknown>,
      }
      setMessages(prev => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        if (last && last.role === 'assistant') {
          updated[updated.length - 1] = {
            ...last,
            traceEvents: [...(last.traceEvents || []), traceEvent],
            isStreaming: true,
          }
        }
        return updated
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
      setMessages(prev => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        if (last && last.role === 'assistant') {
          updated[updated.length - 1] = {
            ...last,
            traceEvents: [...(last.traceEvents || []), traceEvent],
            isStreaming: true,
          }
        }
        return updated
      })
    }
    // `progress` events update the v2 state (already done above) but do
    // not touch messages — the UI's thinking indicator reads from
    // v2StateRef via the status label. `run.paused` is handled by the
    // stream onTerminal callback below.
  }, [])

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
        const message =
          (typeof event.payload.message === 'string' && event.payload.message) ||
          (typeof event.payload.code === 'string' && event.payload.code) ||
          '运行出错'
        const separator = last.content.trim() ? '\n\n' : ''
        return [
          ...updated.slice(0, -1),
          { ...last, content: `${last.content}${separator}[错误] ${message}`, isStreaming: false },
        ]
      }
      // done: just finalize streaming.
      return [...updated.slice(0, -1), { ...last, isStreaming: false }]
    })
    setIsSending(false)
    cleanupRef.current = null
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
    })
    cleanupRef.current = cleanup
  }, [])

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

    try {
      await submitV2Approval(approval.approval_id, decision, decisionRequestId)
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
    const resumeUrl = buildV2ResumeStreamUrl(persisted.run_id, persisted.sequence_cursor)
    attachV2Stream(resumeUrl, '', {
      onOpen: () => setConnectionStatus('connected'),
      onEvent: handleV2Event,
      onTerminal: handleV2Terminal,
      onError: (error) => {
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
  }, [v2ApprovalState, attachV2Stream, handleV2Event, handleV2Terminal])

  // Ticket 08 + 11: refresh recovery (spec AC3 + AC6). On mount, if a v2
  // run was active and has not reached a terminal state, fetch the
  // snapshot and replay events from the stored cursor. This restores
  // the assistant text, tool trace, and — if paused — the pending
  // approval prompt without re-sending the user message.
  useEffect(() => {
    const persisted = v2RunRef.current
    if (!persisted) return
    const persistedRun = persisted
    let cancelled = false

    async function recoverV2Run() {
      try {
        const snapshot = await fetchV2Snapshot(persistedRun.run_id)
        if (cancelled || snapshot === null) {
          // Run unknown on the server (e.g. DB reset). Clear local state.
          v2RunRef.current = null
          saveV2PersistedRun(null)
          setV2ApprovalState('idle')
          return
        }

        // Replay events from the stored cursor. The reducer is idempotent
        // w.r.t. sequence monotonicity, so replaying already-seen events
        // is a no-op.
        const replay = await fetchV2Events(persistedRun.run_id, persistedRun.sequence_cursor)
        if (cancelled) return

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
        const next = reduceV2Events(v2StateRef.current, replayEnvelopes)
        v2StateRef.current = next

        // Restore the user message + assistant placeholder if we have one.
        // We only do this when messages is empty (fresh page load) to avoid
        // duplicating an already-rendered user turn.
        setMessages(prev => {
          if (prev.length > 0) return prev
          const restored: Message[] = []
          if (persistedRun.user_message) {
            restored.push({ role: 'user', content: persistedRun.user_message })
          }
          // Ticket 11: keep the assistant placeholder streaming iff the
          // run is still active (running OR paused). Paused runs render
          // the approve/reject UI on top of the placeholder.
          const isActive = next.status === 'running' || next.status === 'paused'
          if (next.assistantText || isActive) {
            restored.push({
              role: 'assistant',
              content: next.assistantText,
              isStreaming: !isV2Terminal(next),
            })
          }
          return restored
        })

        // Update the persisted cursor so the next recovery starts from here.
        v2RunRef.current = {
          ...persistedRun,
          sequence_cursor: next.sequenceCursor,
          pending_approval: next.pendingApproval,
          v2_status: next.status === 'idle' ? undefined : next.status,
        }
        saveV2PersistedRun(v2RunRef.current)

        if (isV2Terminal(next)) {
          // Terminal already reached: clear the persisted run.
          v2RunRef.current = null
          saveV2PersistedRun(null)
          setV2ApprovalState('idle')
        } else if (isV2Paused(next)) {
          // Ticket 11 spec AC6: paused run — do NOT re-attach a stream.
          // The run is awaiting user approval. The approve/reject UI
          // renders from ``next.pendingApproval``; the user's decision
          // drives the resume flow via handleV2Approval.
          setV2ApprovalState('idle')
        } else {
          // Still running: re-attach a streaming connection to continue
          // receiving events. The server returns the existing run
          // idempotently via request_id.
          const body = JSON.stringify({
            message: persistedRun.user_message,
            request_id: persistedRun.request_id,
            conversation_id: persistedRun.conversation_id,
            profile_id: 'chat',
          })
          attachV2Stream(API.chatV2SSE, body, {
            onOpen: () => setConnectionStatus('connected'),
            onEvent: handleV2Event,
            onTerminal: handleV2Terminal,
            onError: (error) => {
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
              setIsSending(false)
            },
          })
        }
      } catch {
        // Recovery failed (network/snapshot error). Leave the persisted
        // run in place; the user can retry by refreshing again.
      }
    }

    void recoverV2Run()
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [v2Mode, handleV2Event, handleV2Terminal, attachV2Stream])

  // Persist
  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, serializeMessages(messages)) } catch { /* ignore */ }
  }, [messages])

  useEffect(() => {
    if (messages.length > 0) return
    if (v2RunRef.current) return

    let cancelled = false
    const chatId = chatIdRef.current

    async function hydrateServerHistory() {
      try {
        const response = await fetch(API.chatHistory(chatId), {
          method: 'GET',
          headers: { Accept: 'application/json' },
        })
        if (!response.ok) return
        const restored = normalizeServerMessages(await response.json())
        if (restored.length === 0 || cancelled) return
        setMessages(prev => (prev.length === 0 ? restored : prev))
      } catch {
        // Local history remains the fallback.
      }
    }

    void hydrateServerHistory()

    return () => {
      cancelled = true
    }
  }, [])

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
        return prev.map(s => s.id === activeSessionId ? { ...s, chatId: chatIdRef.current, title, date: getTodayDate(), messages, runtime: v2Mode ? 'v2' as const : 'v1' as const } : s)
      }
      return [{ id: activeSessionId, chatId: chatIdRef.current, title, date: getTodayDate(), messages, runtime: v2Mode ? 'v2' as const : 'v1' as const }, ...prev].slice(0, 50)
    })
  }, [activeSessionId, messages, v2Mode])

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => { scrollToBottom() }, [messages, scrollToBottom])

  useEffect(() => { return () => { cleanupRef.current?.() } }, [])

  // Health check
  const checkHealth = useCallback(async () => {
    try {
      const resp = await fetch(API.chatHealth, { method: 'GET' })
      if (resp.ok) setConnectionStatus('connected')
      else setConnectionStatus('error')
    } catch { setConnectionStatus('error') }
  }, [])

  useEffect(() => {
    checkHealth()
    const id = setInterval(checkHealth, 30000)
    return () => clearInterval(id)
  }, [checkHealth])

  const appendToLastMessage = useCallback((text: string) => {
    setMessages(prev => {
      const updated = [...prev]
      const last = updated[updated.length - 1]
      if (last && last.role === 'assistant') {
        updated[updated.length - 1] = { ...last, content: last.content + text, isStreaming: true }
      }
      return updated
    })
  }, [])

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

  const textStream = useMemo(
    () => createTypewriterStreamQueue(appendToLastMessage, { chunkSize: 3, intervalMs: 14 }),
    [appendToLastMessage],
  )

  useEffect(() => {
    return () => textStream.reset()
  }, [textStream])

  const appendTraceEvent = useCallback((event: SSEEvent) => {
    setMessages(prev => {
      const updated = [...prev]
      const last = updated[updated.length - 1]
      if (last && last.role === 'assistant') {
        updated[updated.length - 1] = {
          ...last,
          traceEvents: [...(last.traceEvents || []), {
            type: event.type,
            content: event.content,
            metadata: event.metadata,
          }],
          isStreaming: true,
        }
      }
      return updated
    })
  }, [])

  const appendArtifact = useCallback((event: SSEEvent) => {
    setMessages(prev => {
      const updated = [...prev]
      const last = updated[updated.length - 1]
      if (last && last.role === 'assistant') {
        updated[updated.length - 1] = {
          ...last,
          artifacts: [...(last.artifacts || []), {
            content: event.content,
            metadata: event.metadata,
          }],
          isStreaming: true,
        }
      }
      return updated
    })
  }, [])

  const updateConfirmationTrace = useCallback((confirmationId: string, metadataPatch: Record<string, unknown>) => {
    setMessages(prev => prev.map(message => {
      if (!message.traceEvents?.length) return message
      let changed = false
      const traceEvents = message.traceEvents.map(event => {
        if (event.metadata?.confirmation_id !== confirmationId) return event
        changed = true
        return {
          ...event,
          metadata: {
            ...(event.metadata || {}),
            ...metadataPatch,
          },
        }
      })
      return changed ? { ...message, traceEvents } : message
    }))
  }, [])

  const appendConfirmedToolResult = useCallback((confirmationId: string, response: ToolConfirmationResponse) => {
    const toolResult = response.tool_result
    if (!toolResult) return

    const toolName = response.tool_name
      || (typeof toolResult.tool_name === 'string' ? toolResult.tool_name : 'tool')
    const resultEvent: AgentTraceEvent = {
      type: 'tool_result',
      content: formatConfirmedToolResult(toolResult),
      metadata: {
        ...toolResult,
        tool: toolName,
        confirmation_id: confirmationId,
        confirmation_status: response.status || 'confirmed',
      },
    }

    setMessages(prev => prev.map(message => {
      const traceEvents = message.traceEvents || []
      const hasConfirmation = traceEvents.some(event => event.metadata?.confirmation_id === confirmationId)
      const alreadyAppended = traceEvents.some(event =>
        event.type === 'tool_result' && event.metadata?.confirmation_id === confirmationId
      )
      if (!hasConfirmation || alreadyAppended) return message
      return { ...message, traceEvents: [...traceEvents, resultEvent] }
    }))
  }, [])

  const handleToolConfirmation = useCallback(async (
    confirmationId: string,
    action: ToolConfirmationAction,
  ) => {
    updateConfirmationTrace(confirmationId, {
      confirmation_status: action === 'confirm' ? 'confirming' : 'rejecting',
      confirmation_action: action,
    })

    try {
      const response = await fetch(API.chatConfirm, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
        },
        body: JSON.stringify({
          confirmation_id: confirmationId,
          action,
        }),
      })
      if (!response.ok) {
        throw new Error(`确认请求失败: ${response.status}`)
      }

      const data = await response.json() as ToolConfirmationResponse
      updateConfirmationTrace(confirmationId, {
        confirmation_status: data.status || action,
        confirmation_action: action,
      })
      if (action === 'confirm') {
        appendConfirmedToolResult(confirmationId, data)
      }
    } catch (error) {
      updateConfirmationTrace(confirmationId, {
        confirmation_status: 'error',
        confirmation_error: error instanceof Error ? error.message : '确认请求失败',
      })
    }
  }, [appendConfirmedToolResult, updateConfirmationTrace])

  const finalizeStreaming = useCallback(() => {
    setMessages(prev => {
      const updated = [...prev]
      const last = updated[updated.length - 1]
      if (last?.isStreaming) updated[updated.length - 1] = { ...last, isStreaming: false }
      return updated
    })
  }, [])

  const createNewSession = useCallback(() => {
    if (messages.length > 0 && activeSessionId) {
      const firstUserMsg = messages.find(m => m.role === 'user')
      const title = firstUserMsg ? firstUserMsg.content.slice(0, 30) + (firstUserMsg.content.length > 30 ? '...' : '') : '新对话'
      setSessions(prev => {
        const filtered = prev.filter(s => s.id !== activeSessionId)
        return [{ id: activeSessionId, chatId: chatIdRef.current, title, date: getTodayDate(), messages, runtime: v2Mode ? 'v2' as const : 'v1' as const }, ...filtered].slice(0, 50)
      })
    }
    const newId = generateId()
    const next = createChatId()
    setActiveSessionId(newId)
    setMessages([])
    localStorage.setItem(CHAT_ID_KEY, next)
    chatIdRef.current = next
    setV2Mode(false)
    try { localStorage.removeItem(STORAGE_KEY) } catch { /* ignore */ }
  }, [messages, activeSessionId, v2Mode])

  const loadSession = useCallback(async (session: ChatSession) => {
    if (!session.chatId) {
      session = { ...session, chatId: createChatId() }
    }
    if (activeSessionId && messages.length > 0) {
      const firstUserMsg = messages.find(m => m.role === 'user')
      const title = firstUserMsg ? firstUserMsg.content.slice(0, 30) + (firstUserMsg.content.length > 30 ? '...' : '') : '新对话'
      setSessions(prev => {
        const filtered = prev.filter(s => s.id !== activeSessionId)
        const existing = prev.find(s => s.id === activeSessionId)
        if (existing) {
          return [{ ...existing, chatId: chatIdRef.current, title, date: getTodayDate(), messages, runtime: v2Mode ? 'v2' as const : 'v1' as const }, ...filtered].slice(0, 50)
        }
        return [{ id: activeSessionId, chatId: chatIdRef.current, title, date: getTodayDate(), messages, runtime: v2Mode ? 'v2' as const : 'v1' as const }, ...filtered].slice(0, 50)
      })
    }
    setActiveSessionId(session.id)
    chatIdRef.current = session.chatId
    setV2Mode(session.runtime === 'v2')
    localStorage.setItem(CHAT_ID_KEY, session.chatId)
    setMessages(session.messages)
    try { localStorage.setItem(STORAGE_KEY, serializeMessages(session.messages)) } catch { /* ignore */ }

    if (session.runtime !== 'v2') {
      try {
        const response = await fetch(API.chatHistory(session.chatId), {
          method: 'GET',
          headers: { Accept: 'application/json' },
        })
        if (!response.ok) return
        const restored = normalizeServerMessages(await response.json())
        if (restored.length === 0) return
        setMessages(restored)
        try { localStorage.setItem(STORAGE_KEY, serializeMessages(restored)) } catch { /* ignore */ }
        setSessions(prev => prev.map(s => s.id === session.id ? { ...s, chatId: session.chatId, messages: restored } : s))
      } catch {
        // Local session remains the fallback.
      }
    }
  }, [activeSessionId, messages, v2Mode])

  const deleteSession = useCallback((e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation()
    const session = sessions.find(s => s.id === sessionId)
    if (session?.chatId && session.runtime !== 'v2') {
      void fetch(API.chatHistoryDelete(session.chatId), { method: 'DELETE' }).catch(() => undefined)
    }
    setSessions(prev => prev.filter(s => s.id !== sessionId))
    if (activeSessionId === sessionId) {
      setActiveSessionId(null)
      setMessages([])
      const next = createChatId()
      localStorage.setItem(CHAT_ID_KEY, next)
      chatIdRef.current = next
      setV2Mode(false)
      try { localStorage.removeItem(STORAGE_KEY) } catch { /* ignore */ }
    }
  }, [activeSessionId, sessions])

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
    textStream.reset()

    const now = new Date()
    const timeStr = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`
    const chatId = chatIdRef.current
    const selectedRuntime = await fetchChatRuntime(chatId)
    setV2Mode(selectedRuntime === 'v2')
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

    const finishStreaming = () => {
      finalizeStreaming()
      setIsSending(false)
      cleanupRef.current = null
    }

    // Ticket 08: v2 branch. Spec AC1 requires an independent v2 client and
    // v2 event reducer. v2 does NOT use the v1 typewriter queue — text
    // deltas are written directly to the messages array so v2 stays
    // decoupled from v1 streaming polish. The v2 request body is also
    // strict: only message, request_id, conversation_id, profile_id, and
    // current_location are allowed (server rejects everything else).
    if (selectedRuntime === 'v2') {
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
          onError: (error) => {
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
      return
    }

    try {
      const modelSettings = buildRuntimeModelSettings()
      const payload: Record<string, unknown> = { message: text, chat_id: chatId, scenario: selectedScenario }
      if (Object.keys(modelSettings).length > 0) {
        payload.model_settings = modelSettings
      }
      const needsRouteFollowupLocation = shouldRequestRouteFollowupLocation(text, contextMessages)
      const cachedRouteLocation = needsRouteFollowupLocation ? lastLocationRef.current : null
      const currentLocation = cachedRouteLocation
        || await getBrowserLocation((shouldRequestCurrentLocation(text) || needsRouteFollowupLocation) ? 10000 : 3000)
      if (currentLocation) {
        lastLocationRef.current = currentLocation
        payload.current_location = currentLocation
      } else if (needsRouteFollowupLocation && lastLocationRef.current) {
        payload.current_location = lastLocationRef.current
      }

      const cleanup = createStreamConnection(API.chatSSE, {
        method: 'POST',
        body: JSON.stringify(payload),
        onOpen: () => setConnectionStatus('connected'),
        onMessage: (event: SSEEvent) => {
          if (event.type === 'thought') {
            appendTraceEvent(event)
          } else if (event.type === 'tool_call') {
            appendTraceEvent(event)
          } else if (event.type === 'tool_result') {
            appendTraceEvent(event)
          } else if (event.type === 'approval_required') {
            appendTraceEvent(event)
          } else if (event.type === 'artifact') {
            appendArtifact(event)
          } else if (event.type === 'text') {
            textStream.enqueue(event.content)
          } else if (event.type === 'done') {
            textStream.finishWhenIdle(finishStreaming)
          } else if (event.type === 'error') {
            textStream.flushNow()
            appendToLastMessage(`\n\n[错误] ${event.content}`)
            finishStreaming()
          }
        },
        onError: (error: string) => {
          textStream.flushNow()
          appendToLastMessage(`\n\n[连接错误] ${error}`)
          finishStreaming()
          setConnectionStatus('error')
        },
        onDone: () => {
          textStream.finishWhenIdle(finishStreaming)
          setConnectionStatus('connected')
        },
      })
      cleanupRef.current = cleanup
    } catch (error) {
      textStream.flushNow()
      appendToLastMessage(`\n\n[错误] ${error instanceof Error ? error.message : '请求失败'}`)
      finishStreaming()
    }
  }, [input, isSending, activeSessionId, selectedScenario, messages, appendToLastMessage, appendTraceEvent, appendArtifact, finalizeStreaming, textStream, handleV2Event, handleV2Terminal])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleClear = () => {
    const currentChatId = chatIdRef.current
    setMessages([])
    try {
      if (!v2Mode) {
        void fetch(API.chatHistoryDelete(currentChatId), { method: 'DELETE' }).catch(() => undefined)
      }
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

  const runtime = useExternalStoreRuntime({
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
        try {
          await cancelV2Run(persisted.run_id || persisted.request_id, `cancel-${Date.now()}`)
        } catch {
          // Best-effort: if the cancel API is unreachable, we still
          // finalize the UI locally. The server-side run will eventually
          // time out or be observed via /snapshot on next refresh.
        }
        // Update v2 state to cancelled locally so the UI shows the right
        // status even if the terminal event never arrives.
        v2StateRef.current = { ...v2StateRef.current, status: 'cancelled' }
        v2RunRef.current = null
        saveV2PersistedRun(null)
        appendCancellationNotice()
      } else {
        textStream.flushNow();
        appendCancellationNotice();
      }
      setIsSending(false)
    }
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="flex h-full w-full overflow-hidden relative">
        {/* Floating Sidebar (Drawer) */}
        <FloatingSidebar
          isOpen={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          sessions={sessions}
          activeSessionId={activeSessionId}
          onNewSession={createNewSession}
          onLoadSession={loadSession}
          onDeleteSession={deleteSession}
        />

        {/* Main Gemini Thread Area */}
        <div className="flex-1 flex flex-col relative overflow-hidden bg-transparent">
          {/* Header Toggle Button */}
          {!sidebarOpen && (
            <button
              onClick={() => setSidebarOpen(true)}
              className="hidden md:flex absolute top-6 left-6 z-40 w-10 h-10 items-center justify-center rounded-full bg-black/40 backdrop-blur-md border border-white/10 text-white/80 hover:text-white hover:bg-black/50 transition-colors duration-150 ease-[var(--ease-out)] shadow-sm press-scale"
              title="展开历史对话"
            >
              <Menu className="w-5 h-5" />
            </button>
          )}
          {v2Mode && (
            <div className="absolute top-4 right-4 z-40 flex h-9 items-center gap-3 rounded-md border border-white/10 bg-black/50 px-3 text-xs text-white/80 backdrop-blur-md">
              <label className="flex cursor-pointer items-center gap-2">
                <span>长期记忆</span>
                <input
                  type="checkbox"
                  className="h-4 w-4 accent-emerald-500"
                  checked={memoryEnabled}
                  disabled={memoryPending}
                  onChange={() => { void handleMemoryToggle() }}
                />
              </label>
              <button
                type="button"
                className="flex h-7 w-7 items-center justify-center text-white/70 hover:text-white disabled:opacity-50"
                title="管理长期记忆"
                disabled={!memoryEnabled}
                onClick={() => setMemoryPanelOpen(open => !open)}
              >
                <Database className="h-4 w-4" />
              </button>
            </div>
          )}
          {v2Mode && pendingMemoryConsent && (
            <div className="absolute left-4 right-4 top-16 z-50 ml-auto max-w-sm rounded-md border border-amber-400/30 bg-zinc-950 p-4 text-sm text-white shadow-xl">
              <div className="font-medium">确认保存 {pendingMemoryConsent.category}</div>
              <div className="mt-2 break-words text-white/75">{pendingMemoryConsent.preview}</div>
              <div className="mt-3 flex justify-end gap-2">
                <button className="px-3 py-1.5 text-white/70 hover:text-white" disabled={memoryDecisionPending} onClick={() => { void handleMemoryDecision('reject') }}>拒绝</button>
                <button className="rounded-md bg-emerald-600 px-3 py-1.5 text-white hover:bg-emerald-500 disabled:opacity-50" disabled={memoryDecisionPending} onClick={() => { void handleMemoryDecision('approve') }}>保存</button>
              </div>
            </div>
          )}
          {v2Mode && memoryPanelOpen && (
            <div className="absolute right-4 top-16 z-50 w-[calc(100%-2rem)] max-w-sm rounded-md border border-white/10 bg-zinc-950 p-4 text-sm text-white shadow-xl">
              <div className="flex items-center justify-between">
                <h2 className="font-medium">长期记忆</h2>
                <div className="flex items-center gap-1">
                  <button className="flex h-8 w-8 items-center justify-center text-white/70 hover:text-white" title="导出 JSON" onClick={() => { void handleMemoryExport() }}><Download className="h-4 w-4" /></button>
                  <button className="flex h-8 w-8 items-center justify-center text-red-300 hover:text-red-200" title="清除全部" onClick={() => { void handleMemoryClear() }}><Trash2 className="h-4 w-4" /></button>
                  <button className="flex h-8 w-8 items-center justify-center text-white/70 hover:text-white" title="关闭" onClick={() => setMemoryPanelOpen(false)}><X className="h-4 w-4" /></button>
                </div>
              </div>
              {memoryError && <div className="mt-2 text-xs text-red-300">{memoryError}</div>}
              <div className="mt-3 max-h-72 overflow-y-auto">
                {memoryListPending && <div className="py-6 text-center text-white/50">加载中...</div>}
                {!memoryListPending && memoryItems.length === 0 && <div className="py-6 text-center text-white/50">暂无长期记忆</div>}
                {memoryItems.map(item => (
                  <div key={item.memory_id} className="border-t border-white/10 py-3 first:border-t-0">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="text-xs text-emerald-300">{item.category}</div>
                        <div className="mt-1 break-words text-white/85">{item.content}</div>
                      </div>
                      <div className="flex shrink-0 gap-1">
                        <button className="flex h-7 w-7 items-center justify-center text-white/60 hover:text-white" title="编辑" onClick={() => { void handleMemoryEdit(item) }}><Pencil className="h-3.5 w-3.5" /></button>
                        <button className="flex h-7 w-7 items-center justify-center text-red-300 hover:text-red-200" title="删除" onClick={() => { void handleMemoryDelete(item.memory_id) }}><Trash2 className="h-3.5 w-3.5" /></button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              {memoryTotal > 20 && (
                <div className="mt-3 flex items-center justify-between border-t border-white/10 pt-3 text-xs text-white/60">
                  <button disabled={memoryOffset === 0} onClick={() => setMemoryOffset(Math.max(0, memoryOffset - 20))}>上一页</button>
                  <span>{memoryOffset + 1}-{Math.min(memoryOffset + 20, memoryTotal)} / {memoryTotal}</span>
                  <button disabled={memoryOffset + 20 >= memoryTotal} onClick={() => setMemoryOffset(memoryOffset + 20)}>下一页</button>
                </div>
              )}
            </div>
          )}
          <GeminiThread
            emptyTitle="告诉我你的徒步计划，我将为你规划路线、安排任务，并提供安全建议。"
            onRegenerate={handleRegenerate}
            onConfirmTool={handleToolConfirmation}
            realMessageCount={messages.length}
          />
        </div>
      </div>
    </AssistantRuntimeProvider>
  )
}

export default SuperAgent
