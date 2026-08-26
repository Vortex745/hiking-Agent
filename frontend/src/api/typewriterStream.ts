export interface TypewriterStreamQueue {
  enqueue: (text: string) => void
  finishWhenIdle: (callback: () => void) => void
  flushNow: () => void
  reset: () => void
}

interface TypewriterStreamOptions {
  chunkSize?: number
  intervalMs?: number
  setTimer?: typeof setTimeout
  clearTimer?: typeof clearTimeout
}

export function createTypewriterStreamQueue(
  appendText: (text: string) => void,
  options: TypewriterStreamOptions = {},
): TypewriterStreamQueue {
  const chunkSize = Math.max(1, options.chunkSize ?? 4)
  const intervalMs = Math.max(0, options.intervalMs ?? 18)
  const setTimer = options.setTimer ?? setTimeout
  const clearTimer = options.clearTimer ?? clearTimeout
  let queue = ''
  let timer: ReturnType<typeof setTimeout> | null = null
  let pendingFinish: (() => void) | null = null

  const runPendingFinish = () => {
    if (!pendingFinish) return
    const callback = pendingFinish
    pendingFinish = null
    callback()
  }

  const clearActiveTimer = () => {
    if (timer === null) return
    clearTimer(timer)
    timer = null
  }

  const tick = () => {
    timer = null
    if (!queue) {
      runPendingFinish()
      return
    }

    const next = queue.slice(0, chunkSize)
    queue = queue.slice(next.length)
    appendText(next)

    if (queue) {
      timer = setTimer(tick, intervalMs)
      return
    }

    runPendingFinish()
  }

  const schedule = () => {
    if (timer !== null || !queue) return
    timer = setTimer(tick, intervalMs)
  }

  return {
    enqueue(text: string) {
      if (!text) return
      queue += text
      schedule()
    },
    finishWhenIdle(callback: () => void) {
      if (!queue && timer === null) {
        callback()
        return
      }
      pendingFinish = callback
    },
    flushNow() {
      clearActiveTimer()
      if (queue) {
        appendText(queue)
        queue = ''
      }
      runPendingFinish()
    },
    reset() {
      clearActiveTimer()
      queue = ''
      pendingFinish = null
    },
  }
}
