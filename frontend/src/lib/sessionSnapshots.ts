import { storageKey } from './userStorage'

const SESSIONS_KEY = 'ai-hiking-agent-sessions'
const SESSION_LIMIT = 50
const SENSITIVE_LOCATION_FIELDS = new Set(['latitude', 'longitude', 'accuracy'])

export type AgentTraceEvent = {
  type: string
  content: string
  metadata?: Record<string, unknown>
}

// A02: v2 artifact shape (payload of artifact.available). Download goes
// through /api/v2/artifacts/{artifact_id} — no v1 download_url exists.
export type AgentArtifact = {
  artifact_id: string
  public_filename: string
  mime_type: string
  size_bytes: number
  sha256?: string
  expires_at?: string
  // Ticket 10: the plan book's version relation, verbatim from
  // ``artifact.available`` (``readArtifact`` keeps it only when the server sent
  // a positive integer revision). Absent for anything that is not a generation.
  plan_revision?: number
  previous_artifact_id?: string
}

export interface Message {
  role: 'user' | 'assistant'
  content: string
  time?: string
  isStreaming?: boolean
  /** Runtime-only marker so a fast SSE response still gets the word reveal. */
  hasStreamed?: boolean
  traceEvents?: AgentTraceEvent[]
  artifacts?: AgentArtifact[]
}

export interface ChatSession {
  id: string
  chatId: string
  title: string
  date: string
  messages: Message[]
  snapshotId?: string
  savedAt?: string
}

export type SnapshotReason = 'new_session' | 'switch' | 'window_exit' | 'tab_hidden'

export type SaveSessionsResult = {
  ok: boolean
  reason?: 'quota_exceeded' | 'storage_unavailable'
}

export function createChatId() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return `chat-${crypto.randomUUID()}`
  }
  return `chat-${Date.now()}-${Math.random().toString(36).slice(2)}`
}

export function generateId() {
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`
}

export function getTodayDate() {
  const now = new Date()
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`
}

function createSnapshotId() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return `snap-${crypto.randomUUID()}`
  }
  return `snap-${Date.now()}-${Math.random().toString(36).slice(2)}`
}

export function sanitizeStoredValue(value: unknown): unknown {
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

export function stripTransientMessageState(message: Message): Message {
  const { hasStreamed: _hasStreamed, ...stored } = message
  return stored
}

export function restoreStoredMessages(value: unknown): Message[] {
  if (!Array.isArray(value)) return []
  return value
    .filter((message): message is Message => Boolean(message && typeof message === 'object'))
    .map(stripTransientMessageState)
}

export function snapshotMessages(messages: Message[]): Message[] {
  const frozen = messages.map(message => ({ ...stripTransientMessageState(message), isStreaming: false }))
  return sanitizeStoredValue(frozen) as Message[]
}

function isSessionRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function normalizeSession(raw: Record<string, unknown>): ChatSession {
  const session: ChatSession = {
    id: typeof raw.id === 'string' && raw.id ? raw.id : generateId(),
    chatId: typeof raw.chatId === 'string' && raw.chatId ? raw.chatId : createChatId(),
    title: typeof raw.title === 'string' && raw.title ? raw.title : '新对话',
    date: typeof raw.date === 'string' && raw.date ? raw.date : getTodayDate(),
    messages: restoreStoredMessages(raw.messages),
  }
  if (typeof raw.snapshotId === 'string' && raw.snapshotId) session.snapshotId = raw.snapshotId
  if (typeof raw.savedAt === 'string' && raw.savedAt) session.savedAt = raw.savedAt
  return session
}

function savedAtOf(session: ChatSession): string {
  return session.savedAt ?? ''
}

export type PruneResult = {
  sessions: ChatSession[]
  destroyedCount: number
  firstRemovedId: string | null
}

export function pruneSessions(sessions: unknown): PruneResult {
  if (!Array.isArray(sessions)) return { sessions: [], destroyedCount: 0, firstRemovedId: null }
  let destroyedCount = 0
  let firstRemovedId: string | null = null
  const noteRemoved = (id: unknown) => {
    destroyedCount += 1
    if (firstRemovedId === null && typeof id === 'string' && id) firstRemovedId = id
  }
  const order: string[] = []
  const byId = new Map<string, ChatSession>()
  for (const raw of sessions) {
    if (!isSessionRecord(raw) || typeof raw.id !== 'string' || !raw.id) {
      noteRemoved(isSessionRecord(raw) ? raw.id : undefined)
      continue
    }
    const entry = normalizeSession(raw)
    const existing = byId.get(entry.id)
    if (!existing) {
      byId.set(entry.id, entry)
      order.push(entry.id)
      continue
    }
    if (savedAtOf(entry) > savedAtOf(existing)) byId.set(entry.id, entry)
    noteRemoved(entry.id)
  }
  const kept: ChatSession[] = []
  for (const id of order) {
    if (kept.length >= SESSION_LIMIT) {
      noteRemoved(id)
      continue
    }
    kept.push(byId.get(id) as ChatSession)
  }
  return { sessions: kept, destroyedCount, firstRemovedId }
}

export type UpsertSnapshotResult = {
  sessions: ChatSession[]
  entry: ChatSession
  supersededSnapshotId: string | null
  destroyedCount: number
}

export function upsertSnapshot(
  sessions: ChatSession[],
  input: { id: string; chatId: string; title: string; date: string; messages: Message[] },
  _reason: SnapshotReason,
): UpsertSnapshotResult {
  const existing = sessions.find(session => session.id === input.id)
  const entry: ChatSession = {
    id: input.id,
    chatId: input.chatId,
    title: input.title,
    date: input.date,
    messages: snapshotMessages(input.messages),
    snapshotId: createSnapshotId(),
    savedAt: new Date().toISOString(),
  }
  const next = existing
    ? sessions.map(session => (session === existing ? entry : session))
    : [entry, ...sessions]
  const pruned = pruneSessions(next)
  return {
    sessions: pruned.sessions,
    entry,
    supersededSnapshotId: existing?.snapshotId ?? null,
    destroyedCount: pruned.destroyedCount,
  }
}

function isQuotaExceeded(error: unknown): boolean {
  if (typeof DOMException !== 'undefined' && error instanceof DOMException) {
    return (
      error.name === 'QuotaExceededError'
      || error.name === 'NS_ERROR_DOM_QUOTA_REACHED'
      || error.code === 22
    )
  }
  return Boolean(error && typeof error === 'object' && (error as { name?: unknown }).name === 'QuotaExceededError')
}

export function saveSessions(sessions: ChatSession[]): SaveSessionsResult {
  try {
    const stored = sessions.map(session => ({ ...session, messages: snapshotMessages(session.messages) }))
    localStorage.setItem(storageKey(SESSIONS_KEY), JSON.stringify(stored))
    return { ok: true }
  } catch (error) {
    return { ok: false, reason: isQuotaExceeded(error) ? 'quota_exceeded' : 'storage_unavailable' }
  }
}

export function loadSessions(): ChatSession[] {
  try {
    const saved = localStorage.getItem(storageKey(SESSIONS_KEY))
    const parsed = saved ? JSON.parse(saved) : []
    if (!Array.isArray(parsed)) return []
    return parsed.filter(isSessionRecord).map(normalizeSession)
  } catch { return [] }
}

export function snapshotBytes(entry: ChatSession): number {
  try {
    const json = JSON.stringify(entry)
    if (typeof TextEncoder !== 'undefined') return new TextEncoder().encode(json).byteLength
    if (typeof Blob !== 'undefined') return new Blob([json]).size
    return json.length
  } catch { return 0 }
}
