import { API, API_ORIGIN } from './config'

// The config endpoint lives on the same AI-service v1 base as `API` in
// config.ts. `API_ORIGIN` is either the configured base (…/api/v1) or empty,
// in which case the dev proxy serves /api/v1.
const LLM_CONFIG_ENDPOINT = `${API_ORIGIN || '/api/v1'}/llm-config`
const LLM_CONFIG_TEST_ENDPOINT = `${LLM_CONFIG_ENDPOINT}/test`

const DEFAULT_EMBEDDING_DIMENSIONS = 1536

export type ModelUsage = 'llm' | 'embedding' | 'rerank'
export type ConfigSection = ModelUsage | 'decision'
export type LlmSection = ModelUsage
export type ConfigSource = 'system' | 'account'

/** Server-side view of one section; keys are never returned in plain text. */
export interface LlmConfigSection {
  baseUrl: string
  model: string
  apiKeyMasked: string
  apiKeySet: boolean
}

export interface EmbeddingConfigSection extends LlmConfigSection {
  dimensions: number
}

export type DecisionMode = 'off' | 'shadow' | 'enforce'

export interface LlmConfigDecisionSection extends LlmConfigSection {
  mode: DecisionMode
}

export interface LlmConfigSnapshot {
  llm: LlmConfigSection
  embedding: EmbeddingConfigSection
  rerank: LlmConfigSection
  decision: LlmConfigDecisionSection
  updatedAt: string | null
  indexDimensions: number | null
  /** 每段配置来自账号覆盖还是系统默认。 */
  sources: Record<ConfigSection, ConfigSource>
}

export interface LlmConfigSaveResult extends LlmConfigSnapshot {
  reindexRecommended: boolean
}

/** Editable draft; an empty apiKey means "keep the stored key". */
export interface LlmConfigDraftSection {
  baseUrl: string
  model: string
  apiKey: string
}

export interface EmbeddingConfigDraft extends LlmConfigDraftSection {
  dimensions: number | null
}

export interface DecisionConfigDraft extends LlmConfigDraftSection {
  mode: DecisionMode
}

export interface LlmConfigDraft {
  llm: LlmConfigDraftSection
  embedding: EmbeddingConfigDraft
  rerank: LlmConfigDraftSection
  decision: DecisionConfigDraft
}

export interface LlmConfigTestResult {
  latencyMs: number
  modelCount: number
  models: string[]
}

export class LlmConfigError extends Error {
  readonly status: number
  readonly detail: string

  constructor(status: number, detail: string) {
    super(detail)
    this.name = 'LlmConfigError'
    this.status = status
    this.detail = detail
  }
}

export function isLlmConfigError(value: unknown): value is LlmConfigError {
  return value instanceof LlmConfigError
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function asText(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback
}

function asFiniteNumber(value: unknown): number | null {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

function readSection(value: unknown): LlmConfigSection {
  const record = isRecord(value) ? value : {}
  const apiKeyMasked = asText(record.api_key_masked)
  return {
    baseUrl: asText(record.base_url),
    model: asText(record.model),
    apiKeyMasked,
    apiKeySet: typeof record.api_key_set === 'boolean' ? record.api_key_set : apiKeyMasked.length > 0,
  }
}

function normalizeDecisionMode(value: unknown): DecisionMode {
  return value === 'shadow' || value === 'enforce' ? value : 'off'
}

function normalizeSource(value: unknown): ConfigSource {
  return value === 'account' ? 'account' : 'system'
}

function normalizeSnapshot(value: unknown): LlmConfigSnapshot {
  const root = isRecord(value) ? value : {}
  const embeddingRecord = isRecord(root.embedding) ? root.embedding : {}
  const embedding = readSection(embeddingRecord)
  const dimensions = asFiniteNumber(embeddingRecord.dimensions) ?? DEFAULT_EMBEDDING_DIMENSIONS
  const indexDimensions = asFiniteNumber(root.index_dimensions)
  const decisionRecord = isRecord(root.decision) ? root.decision : {}
  const decision = readSection(decisionRecord)
  const sourcesRecord = isRecord(root.sources) ? root.sources : {}

  return {
    llm: readSection(root.llm),
    embedding: {
      ...embedding,
      dimensions: Math.round(dimensions),
    },
    rerank: readSection(root.rerank),
    decision: {
      ...decision,
      baseUrl: decision.baseUrl || 'https://api.siliconflow.cn/v1',
      model: decision.model || 'Kev-4b',
      mode: normalizeDecisionMode(decisionRecord.mode),
    },
    updatedAt: typeof root.updated_at === 'string' ? root.updated_at : null,
    indexDimensions: indexDimensions === null ? null : Math.round(indexDimensions),
    sources: {
      llm: normalizeSource(sourcesRecord.llm),
      embedding: normalizeSource(sourcesRecord.embedding),
      rerank: normalizeSource(sourcesRecord.rerank),
      decision: normalizeSource(sourcesRecord.decision),
    },
  }
}

/** Build a blank-key draft so password inputs never contain server values. */
export function draftFromSnapshot(snapshot: LlmConfigSnapshot): LlmConfigDraft {
  return {
    llm: {
      baseUrl: snapshot.llm.baseUrl,
      model: snapshot.llm.model,
      apiKey: '',
    },
    embedding: {
      baseUrl: snapshot.embedding.baseUrl,
      model: snapshot.embedding.model,
      dimensions: snapshot.embedding.dimensions,
      apiKey: '',
    },
    rerank: {
      baseUrl: snapshot.rerank.baseUrl,
      model: snapshot.rerank.model,
      apiKey: '',
    },
    decision: {
      baseUrl: snapshot.decision.baseUrl,
      model: snapshot.decision.model,
      mode: snapshot.decision.mode,
      apiKey: '',
    },
  }
}

type FetchLike = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>

function defaultFetch(): FetchLike {
  return globalThis.fetch.bind(globalThis)
}

function readDetail(payload: unknown, fallback: string): string {
  if (!isRecord(payload)) return fallback
  const detail = payload.detail ?? payload.error ?? payload.message
  return typeof detail === 'string' && detail.trim() ? detail.trim() : fallback
}

async function responseError(response: Response): Promise<LlmConfigError> {
  const fallback = `HTTP ${response.status}${response.statusText ? ` ${response.statusText}` : ''}`
  const text = await response.text().catch(() => '')
  if (!text.trim()) return new LlmConfigError(response.status, fallback)
  try {
    return new LlmConfigError(response.status, readDetail(JSON.parse(text), fallback))
  } catch {
    return new LlmConfigError(response.status, text.trim())
  }
}

export async function fetchLlmConfig(
  fetcher: FetchLike = defaultFetch(),
): Promise<LlmConfigSnapshot> {
  const response = await fetcher(LLM_CONFIG_ENDPOINT, {
    method: 'GET',
    headers: { Accept: 'application/json' },
    cache: 'no-store',
  })
  if (!response.ok) throw await responseError(response)
  return normalizeSnapshot(await response.json())
}

function toWireSection(section: LlmConfigDraftSection): Record<string, unknown> {
  const wire: Record<string, unknown> = {
    base_url: section.baseUrl.trim(),
    model: section.model.trim(),
  }
  const apiKey = section.apiKey.trim()
  if (apiKey) wire.api_key = apiKey
  return wire
}

/**
 * Save the editable draft. Empty keys are omitted so the backend keeps the
 * stored secret. The response is the fresh server view plus the reindex hint.
 */
export async function saveLlmConfig(
  draft: LlmConfigDraft,
  fetcher: FetchLike = defaultFetch(),
): Promise<LlmConfigSaveResult> {
  const embedding = toWireSection(draft.embedding)
  if (draft.embedding.dimensions !== null && Number.isFinite(draft.embedding.dimensions)) {
    embedding.dimensions = Math.round(draft.embedding.dimensions)
  }
  const decision = {
    ...toWireSection(draft.decision),
    mode: draft.decision.mode,
  }

  const response = await fetcher(LLM_CONFIG_ENDPOINT, {
    method: 'PUT',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      llm: toWireSection(draft.llm),
      embedding,
      rerank: toWireSection(draft.rerank),
      decision,
    }),
  })
  if (!response.ok) throw await responseError(response)

  const payload: unknown = await response.json()
  return {
    ...normalizeSnapshot(payload),
    reindexRecommended: isRecord(payload) && payload.reindex_recommended === true,
  }
}

/**
 * Reset this account's overrides. The shared system defaults in `.env` are
 * never touched, so another account cannot be affected by this call.
 */
export async function resetLlmConfig(
  fetcher: FetchLike = defaultFetch(),
): Promise<LlmConfigSnapshot> {
  const response = await fetcher(LLM_CONFIG_ENDPOINT, {
    method: 'DELETE',
    headers: { Accept: 'application/json' },
  })
  if (!response.ok) throw await responseError(response)
  return normalizeSnapshot(await response.json())
}

/**
 * Probe one section through the backend. Empty base URL / API key fall back to
 * the stored server values, so a masked key can still be tested.
 */
export async function testLlmConfig(
  section: ConfigSection,
  baseUrl?: string,
  apiKey?: string,
  model?: string,
  fetcher: FetchLike = defaultFetch(),
): Promise<LlmConfigTestResult> {
  const body: Record<string, unknown> = { section }
  if (baseUrl?.trim()) body.base_url = baseUrl.trim()
  if (apiKey?.trim()) body.api_key = apiKey.trim()
  if (model?.trim()) body.model = model.trim()

  const response = await fetcher(LLM_CONFIG_TEST_ENDPOINT, {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  })
  if (!response.ok) throw await responseError(response)

  const payload: unknown = await response.json()
  const latency = isRecord(payload) ? asFiniteNumber(payload.latency_ms) : null
  const modelCount = isRecord(payload) ? asFiniteNumber(payload.model_count) : null
  if (latency === null || modelCount === null) {
    throw new Error('测速响应格式异常')
  }
  const rawModels = isRecord(payload) ? payload.models : null
  const models = Array.isArray(rawModels)
    ? rawModels.filter((id): id is string => typeof id === 'string' && id.trim().length > 0)
    : []
  return {
    latencyMs: Math.max(1, Math.round(latency)),
    modelCount: Math.round(modelCount),
    models,
  }
}

export function maskApiKey(value: string): string {
  if (!value) return '未设置'
  if (value.length < 10) return '已设置'
  return `${value.slice(0, 4)}...${value.slice(-4)}`
}

export interface ModelInfo {
  id: string
  object?: string
  created?: number
  owned_by?: string
  type?: string
  modality?: string
  capabilities?: string[]
  architecture?: {
    modality?: string
    input_modalities?: string[]
    output_modalities?: string[]
    type?: string
  }
}

export interface ModelsResponse {
  object?: string
  data?: ModelInfo[]
  models?: string[]
}

export function filterModelList(models: ModelInfo[], query: string): ModelInfo[] {
  const keyword = query.trim().toLowerCase()
  if (!keyword) return models
  return models.filter(model => model.id.toLowerCase().includes(keyword))
}

const EMBEDDING_HINTS = [
  'embedding',
  'embed',
  'bge',
  'e5',
  'gte',
  'voyage',
  'jina',
  'nomic',
  'text2vec',
]

const RERANK_HINTS = [
  'rerank',
  'reranker',
  'ranker',
  'rank-',
  'rank_',
  'bge-reranker',
]

const LLM_HINTS = [
  'gpt',
  'deepseek',
  'claude',
  'chat',
  'qwen',
  'llama',
  'gemini',
  'glm',
  'doubao',
  'hunyuan',
  'mistral',
  'moonshot',
  'yi-',
]

const NON_CHAT_HINTS = [
  'moderation',
  'whisper',
  'transcribe',
  'transcription',
  'tts',
  'speech',
]

function collectModelHints(model: ModelInfo): string {
  const hintParts = [
    model.id,
    model.object,
    model.owned_by,
    model.type,
    model.modality,
    model.architecture?.modality,
    model.architecture?.type,
    ...(model.capabilities ?? []),
    ...(model.architecture?.input_modalities ?? []),
    ...(model.architecture?.output_modalities ?? []),
  ]

  return hintParts
    .filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
    .join(' ')
    .toLowerCase()
}

type InferredModelUsage = ModelUsage | 'other' | 'unknown'

function inferModelUsageFromHints(hints: string, fallback: 'llm' | 'unknown'): InferredModelUsage {
  if (!hints.trim()) return fallback

  if (RERANK_HINTS.some(keyword => hints.includes(keyword))) {
    return 'rerank'
  }

  if (EMBEDDING_HINTS.some(keyword => hints.includes(keyword))) {
    return 'embedding'
  }

  if (NON_CHAT_HINTS.some(keyword => hints.includes(keyword))) {
    return 'other'
  }

  if (LLM_HINTS.some(keyword => hints.includes(keyword))) {
    return 'llm'
  }

  return fallback
}

function inferModelUsage(model: ModelInfo): ModelUsage | 'other' {
  const hints = collectModelHints(model)
  const inferred = inferModelUsageFromHints(hints, 'llm')
  return inferred === 'unknown' ? 'llm' : inferred
}

export function filterModelsByUsage(
  models: ModelInfo[],
  section: ConfigSection,
): ModelInfo[] {
  if (section === 'decision') {
    const hints = ['jev', 'kev', 'semif', 'noul', 'systemone', 'system-one', 'decision']
    const matched = models.filter(model => {
      const id = model.id.toLowerCase()
      return hints.some(hint => id.includes(hint))
    })
    // A provider may use arbitrary decision-model names; never hide its list.
    return matched.length > 0 ? matched : models
  }
  return models.filter(model => inferModelUsage(model) === section)
}

export function repairModelSelection(
  currentModel: string,
  section: ConfigSection,
  candidates: ModelInfo[],
): string {
  const trimmed = currentModel.trim()
  if (section === 'decision') {
    // Decision aliases are provider-specific; keep a manually entered name.
    if (!candidates.length) return trimmed
    return trimmed || candidates[0].id
  }
  if (!candidates.length) return trimmed
  if (candidates.some(model => model.id === trimmed)) return trimmed

  const inferred = inferModelUsageFromHints(trimmed.toLowerCase(), 'unknown')
  if (!trimmed) return candidates[0].id

  if (inferred !== 'unknown' && inferred !== section) {
    return candidates[0].id
  }

  return trimmed
}

export async function fetchModels(
  baseUrl: string,
  apiKey: string,
  section?: ConfigSection,
  fetcher: FetchLike = defaultFetch(),
): Promise<ModelInfo[]> {
  const body: Record<string, unknown> = { base_url: baseUrl, api_key: apiKey }
  if (section) body.section = section
  const resp = await fetcher(API.modelsFetch, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  if (!resp.ok) {
    const error = await responseError(resp)
    throw new Error(error.message)
  }

  const data: ModelsResponse = await resp.json()
  if (Array.isArray(data.data)) {
    return data.data
  }
  if (Array.isArray(data.models)) {
    return data.models
      .filter((id): id is string => typeof id === 'string' && id.trim().length > 0)
      .map(id => ({ id }))
  }
  return []
}
