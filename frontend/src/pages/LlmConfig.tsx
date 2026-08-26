import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Brain,
  ChevronDown,
  Database,
  Gauge,
  KeyRound,
  ListFilter,
  Loader2,
  RefreshCw,
  RotateCcw,
  Save,
  Server,
  Settings2,
  SlidersHorizontal,
} from 'lucide-react'
import {
  clearLlmSettings,
  DEFAULT_LLM_SETTINGS,
  EmbeddingModelConfig,
  canRunSpeedTest,
  filterModelList,
  filterModelsByUsage,
  fetchModels,
  LlmModelConfig,
  LlmSettings,
  loadLlmSettings,
  maskApiKey,
  measureModelLatency,
  ModelInfo,
  ModelUsage,
  repairModelSelection,
  RerankModelConfig,
  saveLlmSettings,
} from '../api/llmConfig'

type SpeedTestStatus = 'idle' | 'testing' | 'success' | 'error' | 'skipped'

interface SpeedTestState {
  status: SpeedTestStatus
  latencyMs: number | null
  modelCount: number | null
  message: string
  testedAt: string | null
}

function formatTimestamp(value: string | null, fallback: string) {
  if (!value) return fallback
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value))
}

function createIdleSpeedState(): SpeedTestState {
  return {
    status: 'idle',
    latencyMs: null,
    modelCount: null,
    message: '保存后自动测速',
    testedAt: null,
  }
}

function createTestingSpeedState(): SpeedTestState {
  return {
    status: 'testing',
    latencyMs: null,
    modelCount: null,
    message: '测速中...',
    testedAt: null,
  }
}

function createSkippedSpeedState(reason: string): SpeedTestState {
  return {
    status: 'skipped',
    latencyMs: null,
    modelCount: null,
    message: reason,
    testedAt: new Date().toISOString(),
  }
}

function createIdleSpeedTests(): Record<ModelUsage, SpeedTestState> {
  return {
    llm: createIdleSpeedState(),
    embedding: createIdleSpeedState(),
    rerank: createIdleSpeedState(),
  }
}

function createTestingSpeedTests(): Record<ModelUsage, SpeedTestState> {
  return {
    llm: createTestingSpeedState(),
    embedding: createTestingSpeedState(),
    rerank: createTestingSpeedState(),
  }
}

function speedStatusLabel(status: SpeedTestStatus) {
  switch (status) {
    case 'testing':
      return '测速中'
    case 'success':
      return '已完成'
    case 'error':
      return '失败'
    case 'skipped':
      return '已跳过'
    default:
      return '未测速'
  }
}

function usageLabel(usage: ModelUsage): string {
  switch (usage) {
    case 'llm':
      return '大模型'
    case 'embedding':
      return 'Embedding'
    case 'rerank':
      return 'Rerank'
  }
}

function speedStatusClassName(status: SpeedTestStatus) {
  switch (status) {
    case 'testing':
      return 'bg-primary/10 text-primary-light'
    case 'success':
      return 'bg-emerald-500/10 text-emerald-600'
    case 'error':
      return 'bg-red-500/10 text-red-500'
    case 'skipped':
      return 'bg-slate-200 text-slate-600'
    default:
      return 'bg-slate-100 text-text-muted'
  }
}

function ModelCombobox({
  value,
  onChange,
  baseUrl,
  apiKey,
  usage,
}: {
  value: string
  onChange: (v: string) => void
  baseUrl: string
  apiKey: string
  usage: ModelUsage
}) {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    setModels([])
    setError('')
    setOpen(false)
    setQuery('')
  }, [baseUrl, apiKey])

  const handleFetch = useCallback(async () => {
    if (!baseUrl || !apiKey) {
      setError('请先填写 Base URL 和 API Key')
      return
    }
    setLoading(true)
    setError('')
    setModels([])
    setOpen(false)
    try {
      const fetched = await fetchModels(baseUrl, apiKey)
      const filtered = filterModelsByUsage(fetched, usage)
      setModels(filtered)
      setQuery('')
      setOpen(filtered.length > 0)
      const repairedValue = repairModelSelection(value, usage, filtered)
      if (repairedValue && repairedValue !== value) {
        onChange(repairedValue)
      }
      if (fetched.length > 0 && filtered.length === 0) {
        setError(`未识别到可用于 ${usageLabel(usage)} 的候选，请手动输入模型名`)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : '获取失败')
    } finally {
      setLoading(false)
    }
  }, [apiKey, baseUrl, onChange, usage, value])

  const handleSelect = (modelId: string) => {
    setQuery('')
    onChange(modelId)
    setOpen(false)
  }

  const filtered = useMemo(() => filterModelList(models, query), [models, query])

  return (
    <div className="grid gap-1.5">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-text-secondary">模型名称</span>
        <button
          className="inline-flex h-6 items-center gap-1 rounded-sm px-2 text-xs text-text-muted transition-colors duration-150 ease-[var(--ease-out)] hover:bg-primary/10 hover:text-primary-light disabled:cursor-not-allowed disabled:opacity-40"
          disabled={loading}
          onClick={handleFetch}
          type="button"
        >
          {loading ? (
            <Loader2 className="h-3 w-3 animate-spin" />
          ) : (
            <RefreshCw className="h-3 w-3" />
          )}
          {loading ? '获取中...' : '获取模型'}
        </button>
      </div>
      <div className="relative" ref={containerRef}>
        <div className="flex items-center gap-0 rounded-sm border border-border bg-bg-body transition-[border-color,background-color] duration-150 ease-[var(--ease-out)] focus-within:border-primary-light focus-within:bg-white">
          <input
            className="h-10 flex-1 bg-transparent px-3 text-sm outline-none"
            id={`${usage}-model`}
            name={`${usage}-model`}
            value={value}
            onChange={(e) => {
              const nextValue = e.target.value
              setQuery(nextValue)
              onChange(nextValue)
              if (models.length > 0) setOpen(true)
            }}
            onFocus={() => { if (models.length > 0) setOpen(true) }}
          />
          {models.length > 0 && (
            <button
              className="flex h-10 w-8 items-center justify-center text-text-muted hover:text-text-secondary"
              onClick={() => setOpen(prev => !prev)}
              type="button"
            >
              <ChevronDown className={`h-4 w-4 transition-transform duration-200 ${open ? 'rotate-180' : ''}`} />
            </button>
          )}
        </div>
        {error && (
          <div className="mt-1 text-xs text-red-500">{error}</div>
        )}
        {open && filtered.length > 0 && (
          <div className="absolute left-0 right-0 top-full z-10 mt-1 max-h-48 overflow-y-auto rounded-sm border border-border bg-white shadow-lg">
            {filtered.map(model => (
              <button
                className={`block w-full px-3 py-1.5 text-left text-sm transition-colors hover:bg-primary/10 hover:text-primary-light ${model.id === value ? 'bg-primary/5 font-medium text-primary-light' : 'text-text-primary'}`}
                key={model.id}
                onClick={() => handleSelect(model.id)}
                type="button"
              >
                {model.id}
              </button>
            ))}
          </div>
        )}
        {open && filtered.length === 0 && models.length > 0 && (
          <div className="absolute left-0 right-0 top-full z-10 mt-1 rounded-sm border border-border bg-white px-3 py-2 text-xs text-text-muted shadow-lg">
            无匹配模型
          </div>
        )}
      </div>
    </div>
  )
}

function LlmConfig() {
  const [settings, setSettings] = useState<LlmSettings>(() => loadLlmSettings())
  const [notice, setNotice] = useState('待保存')
  const [isAutoTesting, setIsAutoTesting] = useState(false)
  const [speedTests, setSpeedTests] = useState<Record<ModelUsage, SpeedTestState>>(() => createIdleSpeedTests())

  const summary = useMemo(() => [
    { label: '大模型', value: settings.llm.model || DEFAULT_LLM_SETTINGS.llm.model },
    { label: 'Embedding', value: settings.embedding.model || DEFAULT_LLM_SETTINGS.embedding.model },
    { label: 'Rerank', value: settings.rerank.model || DEFAULT_LLM_SETTINGS.rerank.model },
    { label: '密钥', value: maskApiKey(settings.llm.apiKey || settings.embedding.apiKey || settings.rerank.apiKey) },
    { label: '更新时间', value: formatTimestamp(settings.updatedAt, '未保存') },
  ], [settings])

  const runAutoSpeedTests = useCallback(async (savedSettings: LlmSettings) => {
    setIsAutoTesting(true)
    setSpeedTests(createTestingSpeedTests())

    const testedAt = new Date().toISOString()
    const targets = [
      { key: 'llm' as const, config: savedSettings.llm },
      { key: 'embedding' as const, config: savedSettings.embedding },
      { key: 'rerank' as const, config: savedSettings.rerank },
    ]

    const results = await Promise.all(targets.map(async ({ key, config }) => {
      if (!canRunSpeedTest(config.baseUrl, config.apiKey)) {
        return [key, createSkippedSpeedState('缺少 Base URL 或 API Key')] as const
      }

      try {
        const measurement = await measureModelLatency(config.baseUrl, config.apiKey)
        return [key, {
          status: 'success',
          latencyMs: measurement.latencyMs,
          modelCount: measurement.modelCount,
          message: `返回 ${measurement.modelCount} 个模型`,
          testedAt,
        } satisfies SpeedTestState] as const
      } catch (error) {
        return [key, {
          status: 'error',
          latencyMs: null,
          modelCount: null,
          message: error instanceof Error ? error.message : '测速失败',
          testedAt,
        } satisfies SpeedTestState] as const
      }
    }))

    const nextSpeedTests = Object.fromEntries(results) as Record<ModelUsage, SpeedTestState>
    setSpeedTests(nextSpeedTests)
    setIsAutoTesting(false)

    if (Object.values(nextSpeedTests).every(item => item.status === 'skipped')) {
      setNotice('已保存，未执行测速')
      return
    }

    if (Object.values(nextSpeedTests).some(item => item.status === 'error')) {
      setNotice('已保存，测速已完成（含失败项）')
      return
    }

    setNotice('已保存，测速完成')
  }, [])

  const updateLlm = <K extends keyof LlmModelConfig>(field: K, value: LlmModelConfig[K]) => {
    setSettings(prev => ({
      ...prev,
      llm: { ...prev.llm, [field]: value },
      updatedAt: null,
    }))
    setNotice('待保存')
    setSpeedTests(createIdleSpeedTests())
  }

  const updateEmbedding = <K extends keyof EmbeddingModelConfig>(field: K, value: EmbeddingModelConfig[K]) => {
    setSettings(prev => ({
      ...prev,
      embedding: { ...prev.embedding, [field]: value },
      updatedAt: null,
    }))
    setNotice('待保存')
    setSpeedTests(createIdleSpeedTests())
  }

  const updateRerank = <K extends keyof RerankModelConfig>(field: K, value: RerankModelConfig[K]) => {
    setSettings(prev => ({
      ...prev,
      rerank: { ...prev.rerank, [field]: value },
      updatedAt: null,
    }))
    setNotice('待保存')
    setSpeedTests(createIdleSpeedTests())
  }

  const handleSave = async () => {
    const saved = saveLlmSettings(settings)
    setSettings(saved)
    setNotice('已保存，测速中...')
    await runAutoSpeedTests(saved)
  }

  const handleReset = () => {
    const cleared = clearLlmSettings()
    setSettings(cleared)
    setNotice('已重置')
    setIsAutoTesting(false)
    setSpeedTests(createIdleSpeedTests())
  }

  return (
    <div className="h-full overflow-y-auto bg-bg-body px-4 py-4 md:px-8 md:py-6">
      <div className="mx-auto flex w-full max-w-[1180px] flex-col gap-5">
        {/* Header */}
        <section className="t-panel-slide" data-open="true">
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-border pb-5">
            <div className="flex items-center gap-3">
              <div className="t-icon-swap h-11 w-11 rounded-sm bg-primary text-white" data-state="a">
                <span className="t-icon flex h-11 w-11 items-center justify-center" data-icon="a">
                  <Settings2 className="h-5 w-5" />
                </span>
                <span className="t-icon flex h-11 w-11 items-center justify-center" data-icon="b">
                  <SlidersHorizontal className="h-5 w-5" />
                </span>
              </div>
              <div>
                <h1 className="text-[24px] font-bold text-text-primary">LLM 配置</h1>
                <p className="text-sm text-text-secondary">大模型、Embedding 与 Rerank 参数</p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                className="inline-flex h-9 items-center gap-2 rounded-sm border border-border bg-white px-4 text-sm text-text-secondary transition-[transform,border-color,color,background-color] duration-150 ease-[var(--ease-out)] hover:border-primary-light hover:text-primary-light active:scale-[0.97]"
                onClick={handleReset}
              >
                <RotateCcw className="h-4 w-4" />
                重置
              </button>
              <button
                className="inline-flex h-9 items-center gap-2 rounded-sm bg-primary px-4 text-sm text-white transition-[transform,background-color,opacity] duration-150 ease-[var(--ease-out)] hover:bg-primary-hover active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-70"
                disabled={isAutoTesting}
                onClick={handleSave}
              >
                {isAutoTesting ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Save className="h-4 w-4" />
                )}
                {isAutoTesting ? '保存并测速中...' : '保存'}
              </button>
            </div>
          </div>
        </section>

        {/* Summary Cards */}
        <section className="grid grid-cols-5 gap-3 max-xl:grid-cols-3 max-lg:grid-cols-2 max-sm:grid-cols-1">
          {summary.map((item, index) => (
            <div
              className="rounded-sm border border-border bg-white px-4 py-3 shadow-sm"
              key={item.label}
              style={{ animation: `fadeUp 260ms var(--ease-out) ${index * 45}ms both` }}
            >
              <div className="text-xs text-text-muted">{item.label}</div>
              <div className="mt-1 truncate text-sm font-semibold text-text-primary">{item.value}</div>
            </div>
          ))}
        </section>

        <section className="grid grid-cols-3 gap-3 max-lg:grid-cols-1">
          {[
            { key: 'llm' as const, label: '大模型测速', state: speedTests.llm },
            { key: 'embedding' as const, label: 'Embedding 测速', state: speedTests.embedding },
            { key: 'rerank' as const, label: 'Rerank 测速', state: speedTests.rerank },
          ].map((item, index) => (
            <div
              className="rounded-sm border border-border bg-white px-4 py-4 shadow-sm"
              key={item.key}
              style={{ animation: `fadeUp 260ms var(--ease-out) ${index * 55}ms both` }}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-center gap-2">
                  <div className="flex h-9 w-9 items-center justify-center rounded-sm bg-primary/10 text-primary-light">
                    <Gauge className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="text-sm font-semibold text-text-primary">{item.label}</div>
                    <div className="text-xs text-text-muted">
                      {formatTimestamp(item.state.testedAt, '等待保存触发')}
                    </div>
                  </div>
                </div>
                <div className={`rounded-full px-2.5 py-1 text-xs font-medium ${speedStatusClassName(item.state.status)}`}>
                  {speedStatusLabel(item.state.status)}
                </div>
              </div>

              <div className="mt-4 flex items-end justify-between gap-4">
                <div>
                  <div className="text-[24px] font-bold text-text-primary">
                    {item.state.latencyMs === null ? '--' : `${item.state.latencyMs} ms`}
                  </div>
                  <div className="mt-1 text-xs text-text-secondary break-all">{item.state.message}</div>
                </div>
                <div className="text-right text-xs text-text-muted">
                  <div>模型数</div>
                  <div className="mt-1 text-sm font-semibold text-text-primary">
                    {item.state.modelCount === null ? '--' : item.state.modelCount}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </section>

        {/* Model Forms */}
        <section className="grid grid-cols-3 gap-5 max-xl:grid-cols-2 max-lg:grid-cols-1">
          {/* LLM Config */}
          <div className="t-panel-slide rounded-md border border-border bg-white p-5 shadow-sm" data-open="true">
            <div className="mb-5 flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-sm bg-primary/10 text-primary-light">
                <Brain className="h-5 w-5" />
              </div>
              <div>
                <h2 className="text-base font-semibold text-text-primary">大模型配置</h2>
                <div className="text-xs text-text-muted">{notice}</div>
              </div>
            </div>

            <div className="grid gap-4">
              <label className="grid gap-1.5">
                <span className="text-xs font-semibold text-text-secondary">Base URL</span>
                <div className="flex items-center gap-2 rounded-sm border border-border bg-bg-body px-3 transition-[border-color,background-color] duration-150 ease-[var(--ease-out)] focus-within:border-primary-light focus-within:bg-white">
                  <Server className="h-4 w-4 text-text-muted" />
                  <input
                    className="h-10 flex-1 bg-transparent text-sm outline-none"
                    id="llm-base-url"
                    name="llm-base-url"
                    placeholder="https://api.openai.com/v1"
                    value={settings.llm.baseUrl}
                    onChange={(event) => updateLlm('baseUrl', event.target.value)}
                  />
                </div>
              </label>

              <label className="grid gap-1.5">
                <span className="text-xs font-semibold text-text-secondary">API Key</span>
                <div className="flex items-center gap-2 rounded-sm border border-border bg-bg-body px-3 transition-[border-color,background-color] duration-150 ease-[var(--ease-out)] focus-within:border-primary-light focus-within:bg-white">
                  <KeyRound className="h-4 w-4 text-text-muted" />
                  <input
                    className="h-10 flex-1 bg-transparent text-sm outline-none"
                    id="llm-api-key"
                    name="llm-api-key"
                    type="password"
                    value={settings.llm.apiKey}
                    onChange={(event) => updateLlm('apiKey', event.target.value)}
                  />
                </div>
              </label>

              <ModelCombobox
                apiKey={settings.llm.apiKey}
                baseUrl={settings.llm.baseUrl}
                usage="llm"
                value={settings.llm.model}
                onChange={(v) => updateLlm('model', v)}
              />
            </div>
          </div>

          {/* Embedding Config */}
          <div className="t-panel-slide rounded-md border border-border bg-white p-5 shadow-sm" data-open="true">
            <div className="mb-5 flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-sm bg-primary/10 text-primary-light">
                <Database className="h-5 w-5" />
              </div>
              <div>
                <h2 className="text-base font-semibold text-text-primary">Embedding 模型配置</h2>
                <div className="text-xs text-text-muted">{notice}</div>
              </div>
            </div>

            <div className="grid gap-4">
              <label className="grid gap-1.5">
                <span className="text-xs font-semibold text-text-secondary">Base URL</span>
                <div className="flex items-center gap-2 rounded-sm border border-border bg-bg-body px-3 transition-[border-color,background-color] duration-150 ease-[var(--ease-out)] focus-within:border-primary-light focus-within:bg-white">
                  <Server className="h-4 w-4 text-text-muted" />
                  <input
                    className="h-10 flex-1 bg-transparent text-sm outline-none"
                    id="embedding-base-url"
                    name="embedding-base-url"
                    placeholder="https://api.openai.com/v1"
                    value={settings.embedding.baseUrl}
                    onChange={(event) => updateEmbedding('baseUrl', event.target.value)}
                  />
                </div>
              </label>

              <label className="grid gap-1.5">
                <span className="text-xs font-semibold text-text-secondary">API Key</span>
                <div className="flex items-center gap-2 rounded-sm border border-border bg-bg-body px-3 transition-[border-color,background-color] duration-150 ease-[var(--ease-out)] focus-within:border-primary-light focus-within:bg-white">
                  <KeyRound className="h-4 w-4 text-text-muted" />
                  <input
                    className="h-10 flex-1 bg-transparent text-sm outline-none"
                    id="embedding-api-key"
                    name="embedding-api-key"
                    type="password"
                    value={settings.embedding.apiKey}
                    onChange={(event) => updateEmbedding('apiKey', event.target.value)}
                  />
                </div>
              </label>

              <ModelCombobox
                apiKey={settings.embedding.apiKey}
                baseUrl={settings.embedding.baseUrl}
                usage="embedding"
                value={settings.embedding.model}
                onChange={(v) => updateEmbedding('model', v)}
              />

              <label className="grid gap-1.5">
                <span className="text-xs font-semibold text-text-secondary">向量维度</span>
                <input
                  className="h-10 rounded-sm border border-border bg-bg-body px-3 text-sm outline-none transition-[border-color,background-color] duration-150 ease-[var(--ease-out)] focus:border-primary-light focus:bg-white"
                  id="embedding-dimensions"
                  min="64"
                  max="4096"
                  name="embedding-dimensions"
                  step="64"
                  type="number"
                  value={settings.embedding.dimensions}
                  onChange={(event) => updateEmbedding('dimensions', Number(event.target.value))}
                />
              </label>
            </div>
          </div>

          {/* Rerank Config */}
          <div className="t-panel-slide rounded-md border border-border bg-white p-5 shadow-sm" data-open="true">
            <div className="mb-5 flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-sm bg-primary/10 text-primary-light">
                <ListFilter className="h-5 w-5" />
              </div>
              <div>
                <h2 className="text-base font-semibold text-text-primary">Rerank 模型配置</h2>
                <div className="text-xs text-text-muted">{notice}</div>
              </div>
            </div>

            <div className="grid gap-4">
              <label className="grid gap-1.5">
                <span className="text-xs font-semibold text-text-secondary">Base URL</span>
                <div className="flex items-center gap-2 rounded-sm border border-border bg-bg-body px-3 transition-[border-color,background-color] duration-150 ease-[var(--ease-out)] focus-within:border-primary-light focus-within:bg-white">
                  <Server className="h-4 w-4 text-text-muted" />
                  <input
                    className="h-10 flex-1 bg-transparent text-sm outline-none"
                    id="rerank-base-url"
                    name="rerank-base-url"
                    placeholder="https://api.example.com/v1"
                    value={settings.rerank.baseUrl}
                    onChange={(event) => updateRerank('baseUrl', event.target.value)}
                  />
                </div>
              </label>

              <label className="grid gap-1.5">
                <span className="text-xs font-semibold text-text-secondary">API Key</span>
                <div className="flex items-center gap-2 rounded-sm border border-border bg-bg-body px-3 transition-[border-color,background-color] duration-150 ease-[var(--ease-out)] focus-within:border-primary-light focus-within:bg-white">
                  <KeyRound className="h-4 w-4 text-text-muted" />
                  <input
                    className="h-10 flex-1 bg-transparent text-sm outline-none"
                    id="rerank-api-key"
                    name="rerank-api-key"
                    type="password"
                    value={settings.rerank.apiKey}
                    onChange={(event) => updateRerank('apiKey', event.target.value)}
                  />
                </div>
              </label>

              <ModelCombobox
                apiKey={settings.rerank.apiKey}
                baseUrl={settings.rerank.baseUrl}
                usage="rerank"
                value={settings.rerank.model}
                onChange={(v) => updateRerank('model', v)}
              />
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}

export default LlmConfig
