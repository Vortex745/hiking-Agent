import { API } from './config'

export type ModelProvider = 'openai-compatible'

export interface LlmModelConfig {
  provider: ModelProvider
  baseUrl: string
  apiKey: string
  model: string
}

export interface EmbeddingModelConfig {
  provider: ModelProvider
  baseUrl: string
  apiKey: string
  model: string
  dimensions: number
}

export interface RerankModelConfig {
  provider: ModelProvider
  baseUrl: string
  apiKey: string
  model: string
}

export interface LlmSettings {
  llm: LlmModelConfig
  embedding: EmbeddingModelConfig
  rerank: RerankModelConfig
  updatedAt: string | null
}

interface StorageLike {
  getItem(key: string): string | null
  setItem(key: string, value: string): void
  removeItem(key: string): void
}

export const LLM_SETTINGS_STORAGE_KEY = 'ai-hiking-llm-settings'

export const DEFAULT_LLM_SETTINGS: LlmSettings = {
  llm: {
    provider: 'openai-compatible',
    baseUrl: '',
    apiKey: '',
    model: 'deepseek-v4-flash',
  },
  embedding: {
    provider: 'openai-compatible',
    baseUrl: '',
    apiKey: '',
    model: 'text-embedding-3-small',
    dimensions: 1536,
  },
  rerank: {
    provider: 'openai-compatible',
    baseUrl: '',
    apiKey: '',
    model: 'Qwen/Qwen3-Reranker-8B',
  },
  updatedAt: null,
}

const PROVIDERS: ModelProvider[] = ['openai-compatible']

function browserStorage(): StorageLike | null {
  if (typeof window === 'undefined') return null
  return window.localStorage
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function normalizeProvider(value: unknown, fallback: ModelProvider): ModelProvider {
  return typeof value === 'string' && PROVIDERS.includes(value as ModelProvider)
    ? value as ModelProvider
    : fallback
}

function normalizeText(value: unknown, fallback: string): string {
  if (typeof value !== 'string') return fallback
  const trimmed = value.trim()
  return trimmed || fallback
}

function normalizeSecret(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function normalizeNumber(value: unknown, fallback: number, min: number, max: number): number {
  const parsed = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(parsed)) return fallback
  return Math.min(max, Math.max(min, parsed))
}

export function normalizeLlmSettings(value: unknown): LlmSettings {
  const root = isRecord(value) ? value : {}
  const llm = isRecord(root.llm) ? root.llm : {}
  const embedding = isRecord(root.embedding) ? root.embedding : {}
  const rerank = isRecord(root.rerank) ? root.rerank : {}

  return {
    llm: {
      provider: normalizeProvider(llm.provider, DEFAULT_LLM_SETTINGS.llm.provider),
      baseUrl: normalizeText(llm.baseUrl, DEFAULT_LLM_SETTINGS.llm.baseUrl),
      apiKey: normalizeSecret(llm.apiKey),
      model: sanitizeStoredModel(
        normalizeText(llm.model, DEFAULT_LLM_SETTINGS.llm.model),
        'llm',
        DEFAULT_LLM_SETTINGS.llm.model,
      ),
    },
    embedding: {
      provider: normalizeProvider(embedding.provider, DEFAULT_LLM_SETTINGS.embedding.provider),
      baseUrl: normalizeText(embedding.baseUrl, DEFAULT_LLM_SETTINGS.embedding.baseUrl),
      apiKey: normalizeSecret(embedding.apiKey),
      model: sanitizeStoredModel(
        normalizeText(embedding.model, DEFAULT_LLM_SETTINGS.embedding.model),
        'embedding',
        DEFAULT_LLM_SETTINGS.embedding.model,
      ),
      dimensions: Math.round(normalizeNumber(embedding.dimensions, DEFAULT_LLM_SETTINGS.embedding.dimensions, 64, 4096)),
    },
    rerank: {
      provider: normalizeProvider(rerank.provider, DEFAULT_LLM_SETTINGS.rerank.provider),
      baseUrl: normalizeText(rerank.baseUrl, DEFAULT_LLM_SETTINGS.rerank.baseUrl),
      apiKey: normalizeSecret(rerank.apiKey),
      model: sanitizeStoredModel(
        normalizeText(rerank.model, DEFAULT_LLM_SETTINGS.rerank.model),
        'rerank',
        DEFAULT_LLM_SETTINGS.rerank.model,
      ),
    },
    updatedAt: typeof root.updatedAt === 'string' ? root.updatedAt : null,
  }
}

export function loadLlmSettings(storage: StorageLike | null = browserStorage()): LlmSettings {
  if (!storage) return DEFAULT_LLM_SETTINGS

  try {
    const raw = storage.getItem(LLM_SETTINGS_STORAGE_KEY)
    if (!raw) return DEFAULT_LLM_SETTINGS
    return normalizeLlmSettings(JSON.parse(raw))
  } catch {
    return DEFAULT_LLM_SETTINGS
  }
}

export function saveLlmSettings(value: unknown, storage: StorageLike | null = browserStorage()): LlmSettings {
  const settings = {
    ...normalizeLlmSettings(value),
    updatedAt: new Date().toISOString(),
  }

  storage?.setItem(LLM_SETTINGS_STORAGE_KEY, JSON.stringify(settings))
  return settings
}

export function clearLlmSettings(storage: StorageLike | null = browserStorage()): LlmSettings {
  storage?.removeItem(LLM_SETTINGS_STORAGE_KEY)
  return DEFAULT_LLM_SETTINGS
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

export interface SpeedTestMeasurement {
  latencyMs: number
  modelCount: number
}

export type ModelUsage = 'llm' | 'embedding' | 'rerank'
type InferredModelUsage = ModelUsage | 'other' | 'unknown'

export interface RuntimeModelSettingsPayload {
  llm?: {
    base_url: string
    api_key: string
    model: string
  }
  embedding?: {
    base_url: string
    api_key: string
    model: string
    dimensions: number
  }
  rerank?: {
    base_url: string
    api_key: string
    model: string
  }
}

export interface RagQueryPayload {
  question: string
  chat_id?: string
  status: string | null
  model_settings?: RuntimeModelSettingsPayload
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

export function filterModelsByUsage(models: ModelInfo[], usage: ModelUsage): ModelInfo[] {
  return models.filter(model => inferModelUsage(model) === usage)
}

function sanitizeStoredModel(model: string, usage: ModelUsage, fallback: string): string {
  const trimmed = normalizeText(model, fallback)
  const inferred = inferModelUsageFromHints(trimmed.toLowerCase(), 'unknown')

  if (inferred !== 'unknown' && inferred !== usage) {
    return fallback
  }

  return trimmed
}

export function repairModelSelection(
  currentModel: string,
  usage: ModelUsage,
  candidates: ModelInfo[],
): string {
  const trimmed = currentModel.trim()
  if (!candidates.length) return trimmed
  if (candidates.some(model => model.id === trimmed)) return trimmed

  const inferred = inferModelUsageFromHints(trimmed.toLowerCase(), 'unknown')
  if (!trimmed) return candidates[0].id

  if (inferred !== 'unknown' && inferred !== usage) {
    return candidates[0].id
  }

  return trimmed
}

type FetchLike = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>

function defaultFetch(): FetchLike {
  return globalThis.fetch.bind(globalThis)
}

function defaultNow(): number {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now()
  }
  return Date.now()
}

export function canRunSpeedTest(baseUrl: string, apiKey: string): boolean {
  return Boolean(baseUrl.trim() && apiKey.trim())
}

function hasRunnableModelConfig(config: { baseUrl: string; apiKey: string; model: string }): boolean {
  return Boolean(config.baseUrl.trim() && config.apiKey.trim() && config.model.trim())
}

export function buildRuntimeModelSettings(settings: LlmSettings = loadLlmSettings()): RuntimeModelSettingsPayload {
  const payload: RuntimeModelSettingsPayload = {}

  if (hasRunnableModelConfig(settings.llm)) {
    payload.llm = {
      base_url: settings.llm.baseUrl.trim(),
      api_key: settings.llm.apiKey.trim(),
      model: settings.llm.model.trim(),
    }
  }

  if (hasRunnableModelConfig(settings.embedding)) {
    payload.embedding = {
      base_url: settings.embedding.baseUrl.trim(),
      api_key: settings.embedding.apiKey.trim(),
      model: settings.embedding.model.trim(),
      dimensions: settings.embedding.dimensions,
    }
  }

  if (hasRunnableModelConfig(settings.rerank)) {
    payload.rerank = {
      base_url: settings.rerank.baseUrl.trim(),
      api_key: settings.rerank.apiKey.trim(),
      model: settings.rerank.model.trim(),
    }
  }

  return payload
}

export function buildRagQueryPayload(
  question: string,
  status: string | null,
  settings: LlmSettings = loadLlmSettings(),
  chatId?: string,
): RagQueryPayload {
  const modelSettings = buildRuntimeModelSettings(settings)
  const payload: RagQueryPayload = { question, status }
  if (chatId) {
    payload.chat_id = chatId
  }

  if (Object.keys(modelSettings).length > 0) {
    payload.model_settings = modelSettings
  }

  return payload
}

export async function fetchModels(
  baseUrl: string,
  apiKey: string,
  fetcher: FetchLike = defaultFetch(),
): Promise<ModelInfo[]> {
  const resp = await fetcher(API.modelsFetch, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ base_url: baseUrl, api_key: apiKey }),
  })

  if (!resp.ok) {
    const text = await resp.text().catch(() => '')
    let message = `HTTP ${resp.status}`
    try {
      const payload = JSON.parse(text)
      message = payload.detail ?? payload.error ?? message
    } catch {}
    throw new Error(message)
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

export async function measureModelLatency(
  baseUrl: string,
  apiKey: string,
  options: {
    fetcher?: FetchLike
    now?: () => number
  } = {},
): Promise<SpeedTestMeasurement> {
  const fetcher = options.fetcher ?? defaultFetch()
  const now = options.now ?? defaultNow
  const start = now()
  const models = await fetchModels(baseUrl, apiKey, fetcher)
  const end = now()

  return {
    latencyMs: Math.max(1, Math.round(end - start)),
    modelCount: models.length,
  }
}
