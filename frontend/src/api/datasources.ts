import { API_ORIGIN } from './config'

const BASE = `${API_ORIGIN || '/api/v1'}/datasources`

export interface ConnectorAuthField {
  key: string
  label: string
  secret: boolean
  required: boolean
  placeholder: string
  help: string
}

export interface ConnectorSettingField {
  key: string
  label: string
  default: string
  placeholder: string
  help: string
}

export interface ConnectorType {
  type: string
  label: string
  auth_fields: ConnectorAuthField[]
  setting_fields: ConnectorSettingField[]
  supports_tree: boolean
  supports_incremental: boolean
  supports_deletion: boolean
}

export interface SyncCounts {
  discovered: number
  fetched: number
  created: number
  updated: number
  skipped: number
  skipped_by_type: number
  truncated: number
  failed: number
  deleted: number
}

export interface SyncFailure {
  external_id: string
  title: string
  error_code: string
  message: string
}

export interface SyncLog {
  id: string
  datasource_id: string
  trigger: string
  status: 'running' | 'success' | 'partial' | 'failed'
  started_at: string
  finished_at: string | null
  counts: SyncCounts
  failures: SyncFailure[]
  enumeration: Record<string, unknown>
  error_code: string | null
  message: string | null
}

export interface Datasource {
  id: string
  type: string
  name: string
  credentials_configured: Record<string, boolean>
  settings: Record<string, unknown>
  selected_resources: string[]
  schedule: string
  sync_deletions: boolean
  sync_status: 'idle' | 'running'
  running_since: string | null
  next_run_at: string | null
  last_sync_at: string | null
  last_sync_status: string | null
  last_sync: SyncLog | null
  doc_count: number
  created_at: string
  updated_at: string
}

export interface ResourceNode {
  id: string
  title: string
  kind: string
  parent_id: string | null
  has_children: boolean
  rev: string
  url: string
}

export interface DatasourceCreateInput {
  type: string
  name: string
  credentials: Record<string, string>
  settings: Record<string, string>
  selected_resources: string[]
  schedule: string
  sync_deletions: boolean
}

export interface DatasourceUpdateInput {
  name?: string
  credentials?: Record<string, string>
  settings?: Record<string, string>
  selected_resources?: string[]
  schedule?: string
  sync_deletions?: boolean
}

export interface SyncStartResponse {
  sync_id: string
  status: string
}

export interface DocumentEntry {
  external_id: string
  resource_id: string
  rev: string
  rag_source: string
  file_name: string
  last_seen_sync_id: string | null
  updated_at: string
}

export class DatasourceApiError extends Error {
  status: number
  code: string | null

  constructor(message: string, status: number, code: string | null = null) {
    super(message)
    this.name = 'DatasourceApiError'
    this.status = status
    this.code = code
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  const text = await response.text()
  let body: unknown = null
  if (text) {
    try {
      body = JSON.parse(text)
    } catch {
      body = text
    }
  }
  if (!response.ok) {
    const detail = body && typeof body === 'object' ? (body as { detail?: unknown }).detail : null
    const message =
      typeof detail === 'string'
        ? detail
        : detail && typeof detail === 'object' && 'message' in detail
          ? String((detail as { message: unknown }).message)
          : `请求失败（${response.status}）`
    const code =
      detail && typeof detail === 'object' && 'code' in detail
        ? String((detail as { code: unknown }).code)
        : null
    throw new DatasourceApiError(message, response.status, code)
  }
  return body as T
}

export function fetchDatasourceTypes(): Promise<ConnectorType[]> {
  return request<ConnectorType[]>('/types')
}

export function fetchDatasources(): Promise<Datasource[]> {
  return request<Datasource[]>('')
}

export function createDatasource(payload: DatasourceCreateInput): Promise<Datasource> {
  return request<Datasource>('', { method: 'POST', body: JSON.stringify(payload) })
}

export function updateDatasource(id: string, payload: DatasourceUpdateInput): Promise<Datasource> {
  return request<Datasource>(`/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    body: JSON.stringify(payload),
  })
}

export function deleteDatasource(id: string, purgeDocs: boolean): Promise<{ ok: boolean; message: string }> {
  return request(`/${encodeURIComponent(id)}?purge_docs=${purgeDocs ? 'true' : 'false'}`, {
    method: 'DELETE',
  })
}

export function validateCredentials(
  type: string,
  credentials: Record<string, string>,
  settings: Record<string, string>,
): Promise<{ ok: boolean; message: string }> {
  return request('/preview/validate', {
    method: 'POST',
    body: JSON.stringify({ type, credentials, settings }),
  })
}

export function previewResources(
  type: string,
  credentials: Record<string, string>,
  settings: Record<string, string>,
  parentId: string | null,
): Promise<ResourceNode[]> {
  return request('/preview/resources', {
    method: 'POST',
    body: JSON.stringify({ type, credentials, settings, parent_id: parentId }),
  })
}

export function previewAncestors(
  type: string,
  credentials: Record<string, string>,
  settings: Record<string, string>,
  ids: string[],
): Promise<ResourceNode[]> {
  return request('/preview/ancestors', {
    method: 'POST',
    body: JSON.stringify({ type, credentials, settings, ids }),
  })
}

export function validateSavedDatasource(id: string): Promise<{ ok: boolean; message: string }> {
  return request(`/${encodeURIComponent(id)}/validate`, { method: 'POST' })
}

export function fetchSavedResources(id: string, parentId: string | null): Promise<ResourceNode[]> {
  const query = parentId ? `?parent_id=${encodeURIComponent(parentId)}` : ''
  return request(`/${encodeURIComponent(id)}/resources${query}`)
}

export function fetchSavedAncestors(id: string, ids: string[]): Promise<ResourceNode[]> {
  return request(`/${encodeURIComponent(id)}/ancestors?ids=${encodeURIComponent(ids.join(','))}`)
}

export function startDatasourceSync(id: string): Promise<SyncStartResponse> {
  return request(`/${encodeURIComponent(id)}/sync`, { method: 'POST' })
}

export function fetchDatasourceSyncs(id: string, limit = 20): Promise<SyncLog[]> {
  return request(`/${encodeURIComponent(id)}/syncs?limit=${limit}`)
}

export function fetchDatasourceSync(id: string, syncId: string): Promise<SyncLog> {
  return request(`/${encodeURIComponent(id)}/syncs/${encodeURIComponent(syncId)}`)
}

export function fetchDatasourceDocuments(id: string): Promise<DocumentEntry[]> {
  return request(`/${encodeURIComponent(id)}/documents`)
}
