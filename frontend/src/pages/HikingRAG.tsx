import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
import { MessageSquare, Trash2, Menu, X } from 'lucide-react'
import { API } from '../api/config'
import { createStreamConnection, SSEEvent } from '../api/sse'
import { createTypewriterStreamQueue } from '../api/typewriterStream'
import { buildRagQueryPayload } from '../api/llmConfig'
import { applyRagStreamEvent, RagSearchSummary } from '../api/ragStream'
import ConversationMemoryMeter from '../components/ConversationMemoryMeter'
import { useExternalStoreRuntime, AssistantRuntimeProvider, ThreadMessage } from "@assistant-ui/react"
import { GeminiThread } from '../components/assistant-ui/gemini/GeminiThread'
import { FloatingSidebar } from '../components/FloatingSidebar'

interface Message {
  role: 'user' | 'assistant'
  content: string
  time?: string
  isStreaming?: boolean
  processSteps?: string[]
  searchSummary?: RagSearchSummary | null
}

interface ChatSession {
  id: string
  chatId: string
  title: string
  date: string
  messages: Message[]
}

const STORAGE_KEY = 'ai-hiking-rag-chat'
const SESSIONS_KEY = 'ai-hiking-rag-sessions'
const CHAT_ID_KEY = 'ai-hiking-rag-chat-id'
const ACTIVE_SESSION_KEY = 'ai-hiking-rag-active-session'
const USER_CANCELLED_MESSAGE = '用户主动取消回答，请补充内容或点击重试。'

function createChatId() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return `rag-${crypto.randomUUID()}`
  }
  return `rag-${Date.now()}-${Math.random().toString(36).slice(2)}`
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

function generateId() {
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`
}

function getTodayDate() {
  const now = new Date()
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`
}

function sourceLabel(source: string) {
  if (source === 'cloud_docs') return '云知识库'
  if (source === 'upload') return '上传文档'
  if (source === 'unknown') return '未知来源'
  return source
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
  try { localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions)) } catch { /* ignore */ }
}

function normalizeServerMessages(payload: unknown): Message[] {
  const messages = (payload as { messages?: unknown })?.messages
  if (!Array.isArray(messages)) return []
  return messages
    .map(item => {
      const role = (item as { role?: unknown }).role
      const content = (item as { content?: unknown }).content
      if ((role !== 'user' && role !== 'assistant') || typeof content !== 'string') return null
      return { role, content }
    })
    .filter((item): item is Message => item !== null)
}

const convertMessage = (msg: Message, index: number): ThreadMessage => {
  const content: any[] = [{ type: "text", text: msg.content || (msg.isStreaming ? "" : " ") }];

  if (msg.processSteps && msg.processSteps.length > 0) {
    content.push({
      type: "tool-call",
      toolCallId: `step-${index}`,
      toolName: "processSteps",
      args: msg.processSteps
    });
  }
  if (msg.searchSummary) {
    content.push({
      type: "tool-call",
      toolCallId: `search-${index}`,
      toolName: "searchSummary",
      args: msg.searchSummary
    });
  }

  return {
    id: `msg-${index}`,
    role: msg.role,
    content: content as any,
    status: (msg.isStreaming ? { type: "running" } : { type: "complete" }) as any,
    createdAt: new Date(),
    metadata: {},
  } as unknown as ThreadMessage;
};

function HikingRAG() {
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
  const [isSending, setIsSending] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connected' | 'error'>('disconnected')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const cleanupRef = useRef<(() => void) | null>(null)
  const chatIdRef = useRef(getOrCreateChatId())

  // Persist messages
  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(messages)) } catch { /* ignore */ }
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
    if (messages.length > 0) return

    let cancelled = false
    const chatId = chatIdRef.current

    async function hydrateServerHistory() {
      try {
        const response = await fetch(API.ragHistory(chatId), {
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
    return () => { cancelled = true }
  }, [messages.length])

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

  useEffect(() => {
    return () => { cleanupRef.current?.() }
  }, [])

  // Health check
  const checkHealth = useCallback(async () => {
    try {
      const resp = await fetch(API.ragHealth, { method: 'GET' })
      if (resp.ok) setConnectionStatus('connected')
      else setConnectionStatus('error')
    } catch { setConnectionStatus('error') }
  }, [])

  useEffect(() => {
    checkHealth()
    const id = setInterval(checkHealth, 30000)
    return () => clearInterval(id)
  }, [checkHealth])

  const updateLastAssistant = useCallback((updater: (message: Message) => Message) => {
    setMessages(prev => {
      const updated = [...prev]
      const last = updated[updated.length - 1]
      if (last && last.role === 'assistant') {
        updated[updated.length - 1] = updater(last)
      }
      return updated
    })
  }, [])

  const appendToLastMessage = useCallback((text: string) => {
    updateLastAssistant(last => ({ ...last, content: last.content + text, isStreaming: true }))
  }, [updateLastAssistant])

  const appendCancellationNotice = useCallback(() => {
    updateLastAssistant(last => {
      if (last.content.includes(USER_CANCELLED_MESSAGE)) {
        return { ...last, isStreaming: false }
      }
      const separator = last.content.trim() ? '\n\n' : ''
      return {
        ...last,
        content: `${last.content}${separator}${USER_CANCELLED_MESSAGE}`,
        isStreaming: false,
      }
    })
  }, [updateLastAssistant])

  const textStream = useMemo(
    () => createTypewriterStreamQueue(appendToLastMessage, { chunkSize: 3, intervalMs: 14 }),
    [appendToLastMessage],
  )

  useEffect(() => {
    return () => textStream.reset()
  }, [textStream])

  const applyStreamEventToLastMessage = useCallback((event: SSEEvent) => {
    updateLastAssistant(last => {
      const next = applyRagStreamEvent({
        content: last.content,
        processSteps: last.processSteps ?? [],
        searchSummary: last.searchSummary ?? null,
      }, event)

      return {
        ...last,
        content: next.content,
        processSteps: next.processSteps,
        searchSummary: next.searchSummary,
        isStreaming: true,
      }
    })
  }, [updateLastAssistant])

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
  }, [messages, activeSessionId])

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
          return [{ ...existing, chatId: chatIdRef.current, title, date: getTodayDate(), messages }, ...filtered].slice(0, 50)
        }
        return [{ id: activeSessionId, chatId: chatIdRef.current, title, date: getTodayDate(), messages }, ...filtered].slice(0, 50)
      })
    }
    setActiveSessionId(session.id)
    chatIdRef.current = session.chatId
    localStorage.setItem(CHAT_ID_KEY, session.chatId)
    setMessages(session.messages)
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(session.messages)) } catch { /* ignore */ }

    try {
      const response = await fetch(API.ragHistory(session.chatId), {
        method: 'GET',
        headers: { Accept: 'application/json' },
      })
      if (!response.ok) return
      const restored = normalizeServerMessages(await response.json())
      if (restored.length === 0) return
      setMessages(restored)
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify(restored)) } catch { /* ignore */ }
      setSessions(prev => prev.map(s => s.id === session.id ? { ...s, chatId: session.chatId, messages: restored } : s))
    } catch {
      // Local session remains the fallback.
    }
  }, [activeSessionId, messages])

  const deleteSession = useCallback((e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation()
    const session = sessions.find(s => s.id === sessionId)
    if (session?.chatId) {
      void fetch(API.ragHistoryDelete(session.chatId), { method: 'DELETE' }).catch(() => undefined)
    }
    setSessions(prev => prev.filter(s => s.id !== sessionId))
    if (activeSessionId === sessionId) {
      setActiveSessionId(null)
      setMessages([])
      const next = createChatId()
      localStorage.setItem(CHAT_ID_KEY, next)
      chatIdRef.current = next
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
    setIsSending(true)
    textStream.reset()

    const now = new Date()
    const timeStr = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`
    const chatId = chatIdRef.current
    const replaceFromIndex = options.replaceFromIndex
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

    try {
      const cleanup = createStreamConnection(API.ragQuery, {
        method: 'POST',
        body: JSON.stringify(buildRagQueryPayload(text, null, undefined, chatId)),
        onOpen: () => setConnectionStatus('connected'),
        onMessage: (event: SSEEvent) => {
          if (event.type === 'done') {
            textStream.finishWhenIdle(finishStreaming)
          } else if (event.type === 'error') {
            textStream.flushNow()
            appendToLastMessage(`\n\n[错误] ${event.content}`)
            finishStreaming()
          } else if (event.type === 'text') {
            textStream.enqueue(event.content)
          } else {
            applyStreamEventToLastMessage(event)
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
  }, [input, isSending, activeSessionId, appendToLastMessage, applyStreamEventToLastMessage, finalizeStreaming, textStream])

  /** Regenerate: remove the last assistant response, re-send the last user prompt */
  const handleRegenerate = useCallback(() => {
    if (isSending) return
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
      textStream.flushNow();
      appendCancellationNotice();
      setIsSending(false);
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
              className="hidden md:flex absolute top-4 left-4 md:top-6 md:left-6 z-40 w-10 h-10 items-center justify-center rounded-full bg-black/40 backdrop-blur-md border border-white/10 text-white/80 hover:text-white hover:bg-black/50 transition-colors duration-150 ease-[var(--ease-out)] shadow-sm press-scale"
              title="展开历史对话"
            >
              <Menu className="w-5 h-5" />
            </button>
          )}
          <GeminiThread
            emptyTitle="徒步知识问答，有什么我可以帮您？"
            onRegenerate={handleRegenerate}
            realMessageCount={messages.length}
          />
        </div>
      </div>
    </AssistantRuntimeProvider>
  )
}

export default HikingRAG
