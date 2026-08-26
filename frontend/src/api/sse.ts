export interface SSEEvent {
  type: 'thought' | 'process' | 'documents' | 'tool_call' | 'tool_result' | 'approval_required' | 'artifact' | 'text' | 'done' | 'error'
  content: string
  metadata?: Record<string, unknown>
}

export interface StreamHandlers {
  onMessage: (event: SSEEvent) => void
  onError: (error: string) => void
  onOpen?: () => void
  onDone?: () => void
}

export interface CreateStreamOptions {
  method?: string
  body?: string
  onMessage: (event: SSEEvent) => void
  onError: (error: string) => void
  onOpen?: () => void
  onDone?: () => void
}

function parseSSELine(line: string): SSEEvent | null {
  const trimmed = line.trim()
  const jsonStr = trimmed.startsWith('data:')
    ? trimmed.slice(5).trim()
    : trimmed
  if (!jsonStr) return null
  try {
    return JSON.parse(jsonStr) as SSEEvent
  } catch {
    return null
  }
}

export function parseSSEFrame(frame: string): SSEEvent | null {
  const data = frame
    .split(/\r?\n/)
    .map(line => line.trimEnd())
    .filter(line => line.startsWith('data:'))
    .map(line => line.slice(5).trimStart())
    .join('\n')
    .trim()

  if (data) {
    try {
      return JSON.parse(data) as SSEEvent
    } catch {
      return null
    }
  }

  return parseSSELine(frame)
}

async function responseErrorMessage(response: Response) {
  const fallback = `HTTP ${response.status}: ${response.statusText || '请求失败'}`
  const text = await response.text().catch(() => '')
  if (!text.trim()) return fallback

  try {
    const payload = JSON.parse(text) as Record<string, unknown>
    const message = payload.detail ?? payload.error ?? payload.message
    if (typeof message === 'string' && message.trim()) {
      return `HTTP ${response.status}: ${message.trim()}`
    }
  } catch {
    // Plain-text error bodies are useful as-is.
  }

  return `HTTP ${response.status}: ${text.trim()}`
}

export function createStreamConnection(
  url: string,
  options: CreateStreamOptions
): () => void {
  let aborted = false
  let reader: ReadableStreamDefaultReader<Uint8Array> | null = null
  const controller = new AbortController()

  const cleanup = () => {
    aborted = true
    controller.abort()
    reader?.cancel().catch(() => {
      // The stream may already be closed.
    })
  }

  const handleEvent = (event: SSEEvent) => {
    options.onMessage(event)
    if (event.type === 'done') {
      options.onDone?.()
      cleanup()
      return true
    }
    if (event.type === 'error') {
      // error 也是终态事件：消费方（SuperAgent/HikingRAG）在 onMessage
      // 里已按终态处理（finishStreaming）。若不在此终止，服务端关流后
      // 还会触发一次 onError("Stream ended without a terminal event")，
      // 在真实错误之后追加第二条"[连接错误]"。
      cleanup()
      return true
    }
    return false
  }

  const run = async () => {
    try {
      const response = await fetch(url, {
        method: options.method || 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: options.body,
        signal: controller.signal,
      })

      if (!response.ok) {
        options.onError(await responseErrorMessage(response))
        return
      }

      options.onOpen?.()

      reader = response.body?.getReader() ?? null
      if (!reader) {
        options.onError('Response body is not readable')
        return
      }

      const decoder = new TextDecoder()
      let buffer = ''

      while (!aborted) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const frames = buffer.split(/\r?\n\r?\n/)
        buffer = frames.pop() || ''

        for (const frame of frames) {
          if (!frame.trim()) continue
          const event = parseSSEFrame(frame)
          if (event && handleEvent(event)) {
            return
          }
        }
      }

      const remaining = `${buffer}${decoder.decode()}`
      if (remaining.trim() && !aborted) {
        const event = parseSSEFrame(remaining)
        if (event) handleEvent(event)
      }
      if (!aborted) {
        options.onError('Stream ended without a terminal event')
      }
    } catch (error) {
      if (!aborted && !(error instanceof DOMException && error.name === 'AbortError')) {
        options.onError(error instanceof Error ? error.message : 'Connection failed')
      }
    }
  }

  void run()

  return cleanup
}

export function createSSEConnection(
  url: string,
  handlers: StreamHandlers
): () => void {
  let aborted = false
  let retryCount = 0
  const maxRetries = 3

  function connect() {
    if (aborted) return

    const eventSource = new EventSource(url)

    eventSource.onopen = () => {
      retryCount = 0
      handlers.onOpen?.()
    }

    eventSource.onmessage = (event) => {
      const parsed = parseSSELine(event.data)
      if (parsed) {
        handlers.onMessage(parsed)
        if (parsed.type === 'done') {
          handlers.onDone?.()
          eventSource.close()
        }
      }
    }

    eventSource.onerror = () => {
      eventSource.close()
      if (!aborted && retryCount < maxRetries) {
        retryCount++
        const delay = Math.min(1000 * Math.pow(2, retryCount), 10000)
        setTimeout(connect, delay)
      } else if (!aborted) {
        handlers.onError('连接失败，请检查网络后重试')
      }
    }

    return eventSource
  }

  const es = connect()

  return () => {
    aborted = true
    es?.close()
  }
}
