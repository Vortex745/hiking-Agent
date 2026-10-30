const LEGACY_KEYS = [
  'ai-hiking-agent-chat',
  'ai-hiking-agent-sessions',
  'ai-hiking-agent-chat-id',
  'ai-hiking-agent-active-session',
  'ai-hiking-agent-v2-active-run',
]

const CLAIM_MARKER = 'ai-hiking-storage-claimed'

let currentUserId: string | null = null

export function setStorageUser(userId: string | null): void {
  currentUserId = userId
}

export function storageKey(base: string): string {
  return currentUserId ? `u:${currentUserId}:${base}` : base
}

/**
 * 旧匿名会话只搬一次到第一个在本机登录的账号；后登录的账号看不到别人的历史。
 */
export function migrateLegacyStorage(userId: string): void {
  try {
    if (window.localStorage.getItem(CLAIM_MARKER)) return
    for (const key of LEGACY_KEYS) {
      const legacy = window.localStorage.getItem(key)
      if (legacy === null) continue
      const target = storageKey(key)
      if (window.localStorage.getItem(target) === null) {
        window.localStorage.setItem(target, legacy)
      }
    }
    window.localStorage.setItem(CLAIM_MARKER, userId)
  } catch {
    /* localStorage 不可用时不影响登录 */
  }
}
