import { useState, useRef, useEffect, useCallback } from 'react'
import { API } from '../api/config'
import {
  createV2StreamConnection,
  cancelV2Run,
  fetchV2Snapshot,
  fetchV2Events,
  submitV2Approval,
  buildV2ResumeUrlFromDecision,
  fetchV2MemorySettings,
  updateV2MemorySettings,
  submitV2MemoryDecision,
  fetchV2Memories,
  updateV2Memory,
  deleteV2Memory,
  clearV2Memories,
  exportV2Memories,
  planV2Recovery,
  V2_TERMINAL_RUN_STATUSES,
  type V2EventEnvelope,
  type V2ApprovalRequest,
  type V2ApprovalDecisionResponse,
  type V2Memory,
  type V2MemoryCandidate,
  type V2PlanIntent,
  type V2PlanDraftStage,
  type V2PlanDraftState,
  type V2PlanDraftFields,
  type V2PlanRouteReference,
} from '../api/chatV2'
import {
  reduceV2Event,
  reduceV2Events,
  initialV2State,
  isV2Terminal,
  isV2Paused,
  readArtifact as readArtifactFromPayload,
  readPlanDraftState,
  type V2ConversationState,
} from '../api/v2Reducer'
import {
  clearPlanDraft,
  isReopenablePlanDraftStage,
  isSamePlanDraft,
  loadPlanDraft,
  savePlanDraft,
} from '../api/planDraftStore'
import { useExternalStoreRuntime, AssistantRuntimeProvider, ThreadMessage } from "@assistant-ui/react"
import { GeminiThread } from '../components/assistant-ui/gemini/GeminiThread'
import PlanDraftDialog from '../components/PlanDraftDialog'
import Button from '../components/Button'
import { GeminiSidebar } from '../components/assistant-ui/gemini/GeminiSidebar'
import ApprovalCard, { DEFAULT_APPROVAL_QUESTIONS } from '../components/ApprovalCard'
import X from 'reicon-react/icons/X'
import Trash2 from 'reicon-react/icons/Trash2'
import Clipboard from 'reicon-react/icons/Clipboard'
import CheckCircle from 'reicon-react/icons/CheckCircle'
import AlertTriangle from 'reicon-react/icons/AlertTriangle'
import Map from 'reicon-react/icons/Map'
import Download from 'reicon-react/icons/Download'
import Pen from 'reicon-react/icons/Pen'

interface Message {
  role: 'user' | 'assistant'
  content: string
  time?: string
  isStreaming?: boolean
  /** Runtime-only marker so a fast SSE response still gets the word reveal. */
  hasStreamed?: boolean
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
  { icon: CheckCircle, text: '检查装备', scenario: 'gear_check' as AgentScenario },
  { icon: AlertTriangle, text: '风险提醒', scenario: 'risk_assessment' as AgentScenario },
  { icon: Map, text: '推荐路线', scenario: 'route_plan' as AgentScenario },
]

/**
 * Ticket 01: human labels for the whitelisted plan intents. Only used to
 * render the user's own bubble when a send carries structured intent and
 * no typed text — routing never reads these strings; the server reads
 * ``intent``.
 */
const PLAN_INTENT_LABELS: Record<V2PlanIntent, string> = {
  plan_create: '生成行程',
  plan_confirm: '确认生成计划书',
  plan_cancel: '取消生成计划书',
  plan_regenerate: '重新生成草稿',
  plan_select_route: '选用该路线进入计划书草稿',
}

/**
 * Ticket 02: what a restored draft is waiting for. Keyed by the stage union,
 * so TypeScript forces a label for every stage a future ticket adds — the
 * same trick ``PLAN_INTENT_LABELS`` uses to keep the vocabulary from drifting.
 */
const PLAN_STAGE_LABELS: Record<V2PlanDraftStage, string> = {
  awaiting_confirmation: '等你确认是否生成计划书',
  awaiting_input: '等你填写基本信息',
  minimal_draft: '最小草稿已生成，可继续修改',
  cancelled: '已取消',
}

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
  v2_status?: 'running' | 'paused' | 'done' | 'error' | 'cancelled'
  // Ticket 01: the last plan-draft STATE this conversation reached, mirrored
  // from ``plan.draft_state``. ``stage`` is what a later turn (or a refresh)
  // reads to know a draft exists; ``draft`` is what the dialog re-seeds from.
  // This is the localStorage mirror only — ticket 02 owns the restore/revise
  // UX that consumes it.
  plan_draft?: V2PlanDraftState
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
  return JSON.stringify(sanitizeStoredValue(messages.map(stripTransientMessageState)))
}

function stripTransientMessageState(message: Message): Message {
  const { hasStreamed: _hasStreamed, ...stored } = message
  return stored
}

function restoreStoredMessages(value: unknown): Message[] {
  if (!Array.isArray(value)) return []
  return value
    .filter((message): message is Message => Boolean(message && typeof message === 'object'))
    .map(stripTransientMessageState)
}

function sanitizeSessionForStorage(session: ChatSession): ChatSession {
  return {
    ...session,
    messages: sanitizeStoredValue(session.messages.map(stripTransientMessageState)) as Message[],
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
      messages: restoreStoredMessages(session.messages),
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
  const comparisons = new globalThis.Map<string, Record<string, unknown>>();
  for (const event of msg.traceEvents || []) {
    const result = event.metadata?.route_comparison;
    if (result && typeof result === 'object' && typeof (result as Record<string, unknown>).comparison_id === 'string') {
      comparisons.set((result as Record<string, unknown>).comparison_id as string, result as Record<string, unknown>);
    }
  }
  for (const [id, result] of comparisons) {
    content.push({ type: 'tool-call', toolCallId: id, toolName: 'routeComparison', args: result });
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
    metadata: { streamed: msg.hasStreamed === true },
  } as unknown as ThreadMessage;
};

function SuperAgent() {
  const [messages, setMessages] = useState<Message[]>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY)
      return saved ? restoreStoredMessages(JSON.parse(saved)) : []
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
  const [memoryPanelOpen, setMemoryPanelOpen] = useState(false)
  const [memoryItems, setMemoryItems] = useState<V2Memory[]>([])
  const [memoryTotal, setMemoryTotal] = useState(0)
  const [memoryOffset, setMemoryOffset] = useState(0)
  const [memoryListPending, setMemoryListPending] = useState(false)
  const [memoryError, setMemoryError] = useState('')
  const [pendingMemoryConsent, setPendingMemoryConsent] = useState<V2MemoryCandidate | null>(null)
  const [memoryDecisionPending, setMemoryDecisionPending] = useState(false)
  const [pendingApproval, setPendingApproval] = useState<V2ApprovalRequest | null>(null)
  const v2RunRef = useRef<V2PersistedRun | null>(loadV2PersistedRun())
  const v2StateRef = useRef<V2ConversationState>(initialV2State)
  const [v2ApprovalState, setV2ApprovalState] = useState<
    'idle' | 'submitting' | 'resuming' | 'resolved' | 'error'
  >('idle')

  // Ticket 01: plan-book entry state. ``planDraft`` mirrors the server's
  // draft stage (from ``plan.draft_state``); the dialog opens ONLY for
  // ``awaiting_input`` — an ``awaiting_confirmation`` stage renders the
  // confirm prompt instead, because an ambiguous request must be confirmed
  // before the flow opens (spec AC2).
  //
  // Seeded from the persisted run so a refresh mid-run still knows a draft
  // exists; ticket 02 adds the durable half (the conversation-keyed store
  // below), which is what survives a refresh *after* a run terminal. Neither
  // path auto-opens the dialog — restoring shows the 恢复草稿 panel and the
  // user decides, so a page load can never look like an action it did not
  // take (spec AC2 + AC5).
  const [planDraft, setPlanDraft] = useState<V2PlanDraftState | null>(
    () => v2RunRef.current?.plan_draft ?? null,
  )
  const [planDialogOpen, setPlanDialogOpen] = useState(false)

  // StrictMode rule (see the messages-updater comment below): state
  // transitions stay in event handlers, durable writes happen here.
  // Only while a run is persisted — a terminal has already cleared the run
  // record, and resurrecting it would re-trigger stream recovery on reload.
  useEffect(() => {
    if (!planDraft) return
    const persisted = v2RunRef.current
    if (!persisted) return
    if (persisted.plan_draft?.stage === planDraft.stage
      && persisted.plan_draft.draft.destination_or_route === planDraft.draft.destination_or_route
      && persisted.plan_draft.draft.activity_date === planDraft.draft.activity_date
      && persisted.plan_draft.draft.participants === planDraft.draft.participants) {
      return
    }
    v2RunRef.current = { ...persisted, plan_draft: planDraft }
    saveV2PersistedRun(v2RunRef.current)
  }, [planDraft])

  // Ticket 02 AC1/AC2/AC4: the durable mirror, keyed by CONVERSATION rather
  // than by the run. Written from this effect only (never from a `setMessages`
  // updater), and a `cancelled` draft clears the conversation's slot instead
  // of storing it — that is what "取消流程会清理未完成的计划状态" means on the
  // client, and it is why re-entering the conversation shows nothing to
  // restore after a cancel.
  useEffect(() => {
    if (!planDraft) return
    const conversationId = planDraft.conversation_id || chatIdRef.current
    if (!conversationId) return
    if (planDraft.stage === 'cancelled') {
      clearPlanDraft(conversationId)
      return
    }
    savePlanDraft({ ...planDraft, conversation_id: conversationId })
  }, [planDraft])

  // Ticket 02 AC2: restore on entry — page load and conversation switch.
  // ``V2PersistedRun`` is nulled at every run terminal, so the run record
  // cannot be the restore source; this store is. Skipped while a request is
  // in flight, because the live stream owns the draft state then, and the
  // mid-run refresh path already replays ``plan.draft_state`` through
  // ``applyReplay``.
  useEffect(() => {
    if (isSending) return
    const conversationId = chatIdRef.current
    const stored = loadPlanDraft(conversationId)
    // Returning the same object identity when nothing changed keeps this
    // idempotent under StrictMode's double invoke.
    setPlanDraft(current => {
      if (isSamePlanDraft(current, stored)) return current
      if (current && (current.conversation_id || conversationId) === conversationId) return current
      return stored
    })
  }, [activeSessionId, isSending])

  useEffect(() => {
    let cancelled = false
    const ensureMemoryEnabled = async () => {
      try {
        const settings = await fetchV2MemorySettings()
        if (!cancelled && !settings.enabled) await updateV2MemorySettings(true)
      } catch {
        // The memory service fails closed when unavailable; retry on the next mount.
      }
    }
    void ensureMemoryEnabled()
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    if (!memoryPanelOpen) return
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
  }, [memoryPanelOpen, memoryOffset])

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
  // v2 text is written directly to the messages array. The presentation
  // layer owns the word reveal; the reducer stays independent of animation.
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
      setPendingApproval(next.pendingApproval)
    }
    // Ticket 11: ``approval.resolved`` arrives on the resume stream —
    // the decision has been recorded server-side and the run is
    // continuing. Mark the UI as resolved so the approve/reject buttons
    // dismiss gracefully while the next events arrive.
    if (event.type === 'approval.resolved') {
      setV2ApprovalState('resolved')
      setPendingApproval(null)
    }
    if (event.type === 'memory.consent_required') {
      setPendingMemoryConsent(next.pendingMemoryConsent)
    }
    // Ticket 01 AC8: the conversation's plan draft moved stage. Read it from
    // the reducer's validated state, never off the raw payload — an unknown
    // stage keeps the previous state instead of opening a dialog on a guess.
    // ``awaiting_input`` is the ONLY stage that opens the dialog (spec AC1);
    // ``awaiting_confirmation`` must leave it closed until the user confirms
    // (spec AC2), and ``minimal_draft``/``cancelled`` close it.
    if (event.type === 'plan.draft_state') {
      setPlanDraft(next.planDraft)
      setPlanDialogOpen(next.planDraft?.stage === 'awaiting_input')
    }

    if (event.type === 'text') {
      const delta = typeof event.payload.content === 'string' ? event.payload.content : ''
      if (!delta) return
      setMessages(prev => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        if (last && last.role === 'assistant') {
          updated[updated.length - 1] = { ...last, content: last.content + delta, isStreaming: true, hasStreamed: true }
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
      setMessages(prev => {
        const updated = [...prev]
        const last = updated[updated.length - 1]
        if (last && last.role === 'assistant') {
          const separator = last.content.trim() ? '\n\n' : ''
          updated[updated.length - 1] = {
            ...last,
            content: `${last.content}${separator}[注意] ${notice}`,
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
  }, [rotateChatIdAfterForbidden])

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
    const resumeUrl = buildV2ResumeUrlFromDecision(
      decisionResponse ?? { resume_url: '', run_id: persisted.run_id },
      persisted.sequence_cursor,
    )
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

    const applyReplay = async (): Promise<V2ConversationState> => {
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
      const next = reduceV2Events(v2StateRef.current, replayEnvelopes)
      v2StateRef.current = next
      // Ticket 01 AC8: a refresh mid-run replays ``plan.draft_state`` through
      // the pure reducer, which writes the reducer state only — the page's
      // own mirror has to be re-seeded here or the dialog loses the draft it
      // had already opened. Mirroring the stage is all this does: the dialog
      // is NOT auto-reopened from a replay (ticket 02 owns restore UX).
      if (next.planDraft) setPlanDraft(next.planDraft)

      setMessages(prev => {
        if (prev.length === 0) {
          // Fresh page load: restore the user message + assistant turn.
          const restored: Message[] = []
          if (persistedRun.user_message) {
            restored.push({ role: 'user', content: persistedRun.user_message })
          }
          const isActive = next.status === 'running' || next.status === 'paused'
          if (next.assistantText || isActive) {
            restored.push({
              role: 'assistant',
              // Review fix: replayed effect.outcome_unknown events live in
              // reviewNotices (the live path appends them to content
              // directly) — re-append so the replay preserves them.
              content: `${next.assistantText}${v2NoticeSuffix(next.reviewNotices)}`,
              isStreaming: !isV2Terminal(next),
              // Recovered text already existed before this page loaded;
              // only future live deltas may opt into the reveal.
              hasStreamed: false,
              // A02: artifacts delivered before the refresh are part of
              // the replay state — reattach so download cards survive.
              artifacts: next.artifacts.length > 0 ? next.artifacts : undefined,
            })
          }
          return restored
        }
        // Polling replays: refresh the trailing assistant message in
        // place so text/artifacts produced while polling show up when
        // the run finishes.
        const updated = [...prev]
        const last = updated[updated.length - 1]
        if (last && last.role === 'assistant') {
          updated[updated.length - 1] = {
            ...last,
            content: `${next.assistantText}${v2NoticeSuffix(next.reviewNotices)}`,
            isStreaming: !isV2Terminal(next),
            hasStreamed: false,
            artifacts: next.artifacts.length > 0 ? next.artifacts : undefined,
          }
        }
        return updated
      })

      // Update the persisted cursor so the next recovery starts from here.
      v2RunRef.current = {
        ...persistedRun,
        sequence_cursor: next.sequenceCursor,
        pending_approval: next.pendingApproval,
        v2_status: next.status === 'idle' ? undefined : next.status,
      }
      saveV2PersistedRun(v2RunRef.current)
      return next
    }

    const showApprovalPrompt = (next: V2ConversationState) => {
      setV2ApprovalState('idle')
      setPendingApproval(next.pendingApproval)
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
            await applyReplay()
            if (cancelled) return
            clearPersisted()
            return
          }
          if (V2_TERMINAL_RUN_STATUSES.has(snapshot.status)) {
            await applyReplay()
            if (cancelled) return
            clearPersisted()
            return
          }
          if (snapshot.status === 'paused') {
            // Approval became necessary while polling — re-show the
            // prompt; the persisted run stays for handleV2Approval.
            const next = await applyReplay()
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
          await applyReplay()
          if (cancelled) return
          clearPersisted()
          return
        }
        const next = await applyReplay()
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
        const filtered = prev.filter(s => s.id !== activeSessionId)
        return [{ id: activeSessionId, chatId: chatIdRef.current, title, date: getTodayDate(), messages }, ...filtered].slice(0, 50)
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
        const filtered = prev.filter(s => s.id !== activeSessionId)
        const existing = prev.find(s => s.id === activeSessionId)
        if (existing) {
          return [{ ...existing, chatId: chatIdRef.current, title, date: getTodayDate(), messages }, ...filtered].slice(0, 50)
        }
        return [{ id: activeSessionId, chatId: chatIdRef.current, title, date: getTodayDate(), messages }, ...filtered].slice(0, 50)
      })
    }
    setActiveSessionId(target.id)
    chatIdRef.current = target.chatId
    localStorage.setItem(CHAT_ID_KEY, target.chatId)
    const restoredMessages = restoreStoredMessages(target.messages)
    setMessages(restoredMessages)
    try { localStorage.setItem(STORAGE_KEY, serializeMessages(restoredMessages)) } catch { /* ignore */ }
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
    options: {
      replaceFromIndex?: number
      profileId?: 'route_plan'
      /**
       * Ticket 01: a whitelisted structured intent. Plan routing is
       * intent-driven, so an intent send needs no typed text — the old
       * `if (!text) return` guard silently no-oped quick actions.
       */
      intent?: V2PlanIntent
      /** The dialog's validated fields; only sent with plan_create/regenerate. */
      planInput?: V2PlanDraftFields
      /**
       * Ticket 03: the two-id route reference; only sent with plan_select_route.
       * There is no place in this payload for a distance, an ascent or a weather
       * claim — the server re-reads the verified facts from the reference, so the
       * client cannot supply one (spec AC3/AC4).
       */
      routeSelection?: V2PlanRouteReference
    } = {},
  ) => {
    const text = typeof textOverride === 'string' ? textOverride.trim() : input.trim()
    if (isSending) return
    if (!text && !options.intent) return

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
    // An intent send may carry no typed text. ``ChatV2Request.message`` is
    // ``min_length=1``, so a bare intent would 422 at the wire boundary —
    // the intent's human label travels instead. Routing never reads it:
    // the server branches on ``intent`` before the classifier runs.
    const requestText = text || (options.intent ? PLAN_INTENT_LABELS[options.intent] : '')
    const nextUserMessage: Message = {
      role: 'user',
      content: requestText,
      time: timeStr,
    }
    const nextAssistantMessage: Message = { role: 'assistant', content: '', isStreaming: true, hasStreamed: true }

    setMessages(prev => {
      const base = typeof replaceFromIndex === 'number'
        ? prev.slice(0, Math.max(0, replaceFromIndex))
        : prev
      return [...base, nextUserMessage, nextAssistantMessage]
    })

    // Ticket 08: v2 是唯一 chat 链路。Spec AC1 requires an independent v2
    // client and v2 event reducer. Text deltas are written directly to the
    // messages array; the presentation layer handles word reveal. The v2
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
        user_message: requestText,
      }
      saveV2PersistedRun(v2RunRef.current)

      try {
        const v2Payload: Record<string, unknown> = {
          message: requestText,
          request_id: requestId,
          conversation_id: conversationId,
          profile_id: options.profileId || (selectedScenario === 'route_plan' ? 'route_plan' : 'chat'),
        }
        // Ticket 01: the plan entry travels as structured data. The strict
        // server model rejects anything outside its whitelist, so an intent
        // here is the only way in — never a sentence carrying a pseudo id.
        if (options.intent) {
          v2Payload.intent = options.intent
          if (options.planInput) v2Payload.plan_input = options.planInput
          if (options.routeSelection) v2Payload.route_selection = options.routeSelection
        }
        const needsRouteFollowupLocation = shouldRequestRouteFollowupLocation(text, contextMessages)
        // A plan send must not trigger a geolocation prompt: the draft flow
        // is field-driven and profile `chat`.
        const cachedRouteLocation = !options.intent && needsRouteFollowupLocation ? lastLocationRef.current : null
        const currentLocation = options.intent
          ? null
          : cachedRouteLocation
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
    }
  }, [input, isSending, activeSessionId, selectedScenario, messages, handleV2Event, handleV2Terminal])

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
        setPendingApproval(null)
        appendCancellationNotice()
      }
      setIsSending(false)
    }
  });

  // 问答框（ApprovalCard）提交后，把所选选项拼成一句偏好描述，作为一条
  // 用户消息发回对话流，让 agent 基于答案继续规划。
  const handleClarificationSubmitted = useCallback((answers: Record<number, number[]>) => {
    const parts: string[] = []
    DEFAULT_APPROVAL_QUESTIONS.forEach((question, qi) => {
      const picked = answers[qi] ?? []
      if (picked.length > 0) {
        parts.push(`${question.q}${picked.map(i => question.options[i]).join('、')}`)
      }
    })
    if (parts.length > 0) {
      void handleSend(`我的偏好：${parts.join('；')}`)
    }
  }, [handleSend])

  // ---- Ticket 01: plan-book entry sends ------------------------------------
  // Each one is an ordinary conversational turn that carries a whitelisted
  // ``intent`` (seam-contract ruling), NOT the tool-approval pause machinery
  // and NOT prose the server has to re-parse the way
  // ``handleClarificationSubmitted`` above does.
  //
  // They are rebuilt with ``handleSend`` on purpose: ``handleSend`` closes
  // over ``input``/``messages``/``isSending`` and is recreated whenever those
  // change, so a dialog that stays open across several turns must submit
  // through the current closure. Memoising these behind a stable identity (or
  // stashing ``handleSend`` in a ref set once) would hand the long-lived
  // dialog a stale ``isSending`` and silently no-op or double-send.
  const handleGeneratePlan = useCallback(() => {
    // The dialog is opened by the server's ``awaiting_input`` event, not
    // optimistically: a blocked intent must leave it closed.
    void handleSend('', { intent: 'plan_create' })
  }, [handleSend])

  const handlePlanDialogSubmit = useCallback((fields: V2PlanDraftFields) => {
    void handleSend('', { intent: 'plan_create', planInput: fields })
  }, [handleSend])

  // Ticket 03 AC1: pick the recommendation or an alternative from a comparison
  // the thread already shows. The send carries the two-id reference and nothing
  // else — the verified facts come back from the server, re-read through
  // RouteSession, so a click cannot invent a distance or a weather claim (AC4).
  // Profile ``route_plan`` is the widened entry: a plan intent on it joins the
  // plan flow instead of being handed to the kernel.
  const handleSelectRoute = useCallback((selection: V2PlanRouteReference) => {
    void handleSend('', {
      intent: 'plan_select_route',
      routeSelection: selection,
      profileId: 'route_plan',
    })
  }, [handleSend])

  const handlePlanDialogDismiss = useCallback(() => {
    // AC2: closing the dialog parks the draft — nothing leaves the page, so
    // the conversation keeps its stored stage and the 恢复草稿 panel takes
    // over. Cancelling has its own action below; conflating the two would
    // destroy the draft the moment the user stepped away.
    setPlanDialogOpen(false)
  }, [])

  // AC4: the one exit that clears the plan state. Only a stage the user
  // actually reached can be cancelled — cancelling ``awaiting_confirmation``
  // (nothing open) or an already-cancelled draft would come back as
  // ``cancel_without_active_draft``. The shared stage set is also what the
  // restore panel is keyed on, so the two cannot drift.
  const handlePlanDialogCancel = useCallback(() => {
    setPlanDialogOpen(false)
    if (planDraft && isReopenablePlanDraftStage(planDraft.stage)) {
      void handleSend('', { intent: 'plan_cancel' })
    }
  }, [handleSend, planDraft])

  // Ticket 02 AC2: reopen the dialog from the draft this conversation already
  // has. Deliberately local — no request leaves the page, so restoring a
  // draft cannot start a route comparison, a plan judgement, or a PDF (AC5).
  // The form re-seeds from the stored fields, and the read-only summary block
  // from the stored summary, on the dialog's own open effect.
  const handleRestorePlanDraft = useCallback(() => {
    setPlanDialogOpen(true)
  }, [])

  const handlePlanConfirm = useCallback(() => {
    void handleSend('', { intent: 'plan_confirm' })
  }, [handleSend])

  const handlePlanDecline = useCallback(() => {
    void handleSend('', { intent: 'plan_cancel' })
  }, [handleSend])

  return (
    <AssistantRuntimeProvider runtime={runtime}>
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
          {/* 问答框：agent 向用户收集偏好/确认，内嵌在主线程区右上角 */}
          <div className="absolute right-4 top-16 z-40">
            <ApprovalCard
              questions={DEFAULT_APPROVAL_QUESTIONS}
              onSubmitted={handleClarificationSubmitted}
            />
          </div>
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
          {pendingMemoryConsent && (
            <div className="absolute left-4 right-4 top-16 z-50 ml-auto max-w-sm rounded-md border border-amber-400/30 bg-zinc-950 p-4 text-sm text-white shadow-xl">
              <div className="font-medium">确认保存 {pendingMemoryConsent.category}</div>
              <div className="mt-2 break-words text-white/75">{pendingMemoryConsent.preview}</div>
              <div className="mt-3 flex justify-end gap-2">
                <button className="px-3 py-1.5 text-white/70 hover:text-white" disabled={memoryDecisionPending} onClick={() => { void handleMemoryDecision('reject') }}>拒绝</button>
                <button className="rounded-md bg-emerald-600 px-3 py-1.5 text-white hover:bg-emerald-500 disabled:opacity-50" disabled={memoryDecisionPending} onClick={() => { void handleMemoryDecision('approve') }}>保存</button>
              </div>
            </div>
          )}
          {memoryPanelOpen && (
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
                        <button className="flex h-7 w-7 items-center justify-center text-white/60 hover:text-white" title="编辑" onClick={() => { void handleMemoryEdit(item) }}><Pen className="h-3.5 w-3.5" /></button>
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
          {/* Ticket 01 AC2: an ambiguous planning request parks the draft in
              ``awaiting_confirmation`` and the dialog stays shut. This prompt
              is the confirmation affordance; both answers travel as a
              whitelisted intent, never as text the server re-parses. */}
          {planDraft?.stage === 'awaiting_confirmation' && (
            <div className="absolute left-4 right-4 top-16 z-50 ml-auto max-w-sm rounded-md border border-border bg-white p-4 text-sm text-text-primary shadow-xl dark:border-white/10 dark:bg-[#1e1f20]">
              <div className="font-medium">是否生成计划书？</div>
              <p className="mt-1 text-xs text-text-muted">
                你的请求还不够明确。确认后才会打开填写弹窗，未确认前不会生成任何计划内容。
              </p>
              <div className="mt-3 flex justify-end gap-2">
                <Button variant="ghost" size="sm" disabled={isSending} onClick={handlePlanDecline}>
                  暂不生成
                </Button>
                <Button variant="accent" size="sm" disabled={isSending} onClick={handlePlanConfirm}>
                  确认生成
                </Button>
              </div>
            </div>
          )}
          {/* Ticket 02 AC2: the restore affordance. A draft in an open stage
              survives a refresh or a conversation switch (conversation-keyed
              store), and this panel is how the user gets it back — reopening
              is a local action, so nothing is compared, judged, or rendered.
              It stays hidden while the dialog itself is open, and
              ``awaiting_confirmation`` has its own prompt above instead. */}
          {planDraft && !planDialogOpen && isReopenablePlanDraftStage(planDraft.stage) && (
            <div className="absolute left-4 right-4 top-16 z-50 ml-auto max-w-sm rounded-md border border-border bg-white p-4 text-sm text-text-primary shadow-xl dark:border-white/10 dark:bg-[#1e1f20]">
              <div className="font-medium">本会话有未完成的行程草稿</div>
              <p className="mt-1 text-xs text-text-muted">
                草稿 {planDraft.draft_id ? planDraft.draft_id.slice(0, 8) : '—'}：
                {PLAN_STAGE_LABELS[planDraft.stage]}
                {planDraft.draft.destination_or_route
                  ? ` · ${planDraft.draft.destination_or_route}`
                  : ''}
                {planDraft.draft.activity_date ? ` · ${planDraft.draft.activity_date}` : ''}
                {planDraft.draft.participants != null ? ` · ${planDraft.draft.participants} 人` : ''}
              </p>
              <div className="mt-3 flex justify-end gap-2">
                <Button variant="ghost" size="sm" disabled={isSending} onClick={handlePlanDialogCancel}>
                  取消草稿
                </Button>
                <Button variant="accent" size="sm" onClick={handleRestorePlanDraft}>
                  恢复草稿
                </Button>
              </div>
            </div>
          )}
          <GeminiThread
            onExportRoute={comparisonId => { void handleSend(`请导出带来源的路线比较 PDF，comparison_id=${comparisonId}`, { profileId: 'route_plan' }) }}
            onSelectRoute={handleSelectRoute}
            onGeneratePlan={handleGeneratePlan}
            isSending={isSending}
            emptyTitle="告诉我你的徒步计划，我将为你规划路线、安排任务，并提供安全建议。"
            onRegenerate={handleRegenerate}
            realMessageCount={messages.length}
          />
          {/* Ticket 01 AC1/AC6: the entry dialog. Opens only from the
              server's ``awaiting_input`` stage, and re-seeds from the stored
              draft so a second entry does not discard what the user typed.
              Ticket 02: the same stored draft carries its summary, shown
              read-only above the form so an edit and its summary stay in sync. */}
          <PlanDraftDialog
            open={planDialogOpen}
            busy={isSending}
            initial={planDraft?.draft}
            summary={planDraft?.summary}
            onDismiss={handlePlanDialogDismiss}
            onCancel={handlePlanDialogCancel}
            onSubmit={handlePlanDialogSubmit}
          />
        </div>
      </div>
    </AssistantRuntimeProvider>
  )
}

export default SuperAgent
