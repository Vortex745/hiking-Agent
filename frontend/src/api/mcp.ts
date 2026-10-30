import { API_ORIGIN } from './config'

const root = `${API_ORIGIN ? API_ORIGIN.replace(/\/api\/v1\/?$/, '/api/v2') : '/api/v2'}/mcp`
const base = `${root}/servers`

export interface McpServer {
  id: string
  name: string
  builtin: boolean
  transport: 'http' | 'stdio'
  status: 'configured' | 'untested' | 'success' | 'failed'
  tools: string[]
  url?: string
  command?: string
  args?: string[]
  header_names?: string[]
  env_names?: string[]
  enabled?: boolean
}

export interface McpDraft {
  name: string
  transport: 'http' | 'stdio'
  url: string
  command: string
  args: string[]
  headers: Record<string, string> | null
  env: Record<string, string> | null
  enabled: boolean
  test_id?: string
  server_id?: string
  stdio_confirmed?: boolean
}

async function request<T>(url: string, method = 'GET', body?: unknown): Promise<T> {
  const response = await fetch(url, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(typeof error.detail === 'string' ? error.detail : `请求失败 (${response.status})`)
  }
  return response.json() as Promise<T>
}

export interface McpTestSuccess {
  status: 'success'
  test_id: string
  tools: string[]
}

export interface McpAuthRequired {
  status: 'auth_required'
  auth_url: string
  flow_id: string
}

export type McpTestResult = McpTestSuccess | McpAuthRequired

export interface McpOAuthStatus {
  status: 'starting' | 'auth_required' | 'success' | 'error'
  test_id?: string
  tools?: string[]
  detail?: string
}

export const listMcpServers = () => request<{ servers: McpServer[] }>(base)
export const testMcpServer = (draft: McpDraft) =>
  request<McpTestResult>(`${base}/test`, 'POST', draft)
export const getMcpOAuthStatus = (flowId: string) =>
  request<McpOAuthStatus>(`${root}/oauth/status/${encodeURIComponent(flowId)}`)
export const saveMcpServer = (draft: McpDraft, id?: string) =>
  request<McpServer>(id ? `${base}/${encodeURIComponent(id)}` : base, id ? 'PUT' : 'POST', draft)
export const deleteMcpServer = (id: string) =>
  request<{ deleted: boolean }>(`${base}/${encodeURIComponent(id)}`, 'DELETE')
