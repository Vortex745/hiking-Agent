export interface AuthUser {
  user_id: string
  login_id: string
  created_at: string
}

export class UnauthorizedError extends Error {
  constructor() {
    super('unauthorized')
    this.name = 'UnauthorizedError'
  }
}

export class AuthApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.name = 'AuthApiError'
    this.status = status
  }
}

const AUTH_BASE = '/api/auth'

function detailMessage(payload: unknown, fallback: string): string {
  if (payload && typeof payload === 'object') {
    const detail = (payload as { detail?: unknown }).detail
    if (typeof detail === 'string') return detail
    if (detail && typeof detail === 'object') {
      const message = (detail as { message?: unknown }).message
      if (typeof message === 'string') return message
    }
    const message = (payload as { message?: unknown }).message
    if (typeof message === 'string') return message
  }
  return fallback
}

async function request<T>(
  path: string,
  init?: RequestInit,
  options?: { sessionCheck?: boolean },
): Promise<T> {
  const response = await fetch(`${AUTH_BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
  })
  const payload = await response.json().catch(() => null)
  if (response.status === 401) {
    if (options?.sessionCheck) throw new UnauthorizedError()
    throw new AuthApiError(detailMessage(payload, '登录 ID 或密码错误'), 401)
  }
  if (!response.ok) {
    throw new AuthApiError(
      detailMessage(payload, `请求失败（${response.status}）`),
      response.status,
    )
  }
  return payload as T
}

export function getMe(): Promise<AuthUser> {
  return request<AuthUser>('/me', undefined, { sessionCheck: true })
}

export function login(loginId: string, password: string): Promise<AuthUser> {
  return request<AuthUser>('/login', {
    method: 'POST',
    body: JSON.stringify({ login_id: loginId, password }),
  })
}

export function register(loginId: string, password: string): Promise<AuthUser> {
  return request<AuthUser>('/register', {
    method: 'POST',
    body: JSON.stringify({ login_id: loginId, password }),
  })
}

export async function logout(): Promise<void> {
  await request<{ ok: boolean }>('/logout', { method: 'POST' })
}
