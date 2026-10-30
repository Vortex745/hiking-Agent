import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Brain,
  ChevronDown,
  ChevronRight,
  CircleCheck,
  Cpu,
  Database,
  Filter,
  Gauge,
  Info,
  Key,
  LoaderCircle,
  RefreshCw,
  RotateCcw,
  Save,
  Server,
  TriangleAlert,
  Undo2,
} from 'lucide-react'
import ArcCard from '../components/ui/card'
import ArcDialog from '../components/ui/dialog'
import { useAuth } from '../auth/AuthContext'
import {
  ConfigSection,
  DecisionConfigDraft,
  DecisionMode,
  draftFromSnapshot,
  fetchLlmConfig,
  fetchModels,
  filterModelList,
  filterModelsByUsage,
  isLlmConfigError,
  LlmConfigDecisionSection,
  LlmConfigDraft,
  LlmConfigDraftSection,
  LlmConfigSection,
  LlmConfigSnapshot,
  ModelInfo,
  ModelUsage,
  repairModelSelection,
  resetLlmConfig,
  saveLlmConfig,
  testLlmConfig,
} from '../api/llmConfig'

type LoadState = 'loading' | 'ready' | 'unsupported' | 'error'

type SpeedTestStatus = 'idle' | 'testing' | 'success' | 'error'

interface SpeedTestState {
  status: SpeedTestStatus
  latencyMs: number | null
  modelCount: number | null
  message: string
  testedAt: string | null
}

interface SaveFeedback {
  tone: 'success' | 'error'
  title: string
  detail?: string
  conflict?: boolean
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

function apiKeyPlaceholder(section: LlmConfigSection): string {
  return section.apiKeySet ? '已设置，留空保持不变' : '未设置'
}

function sourceLabel(source: LlmConfigSnapshot['sources'][ConfigSection]): string {
  if (source === 'account') return '账号'
  if (source === 'system') return '系统'
  return '未配置'
}

function createIdleSpeedState(): SpeedTestState {
  return {
    status: 'idle',
    latencyMs: null,
    modelCount: null,
    message: '尚未测速',
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

function createIdleSpeedTests(): Record<ConfigSection, SpeedTestState> {
  return {
    llm: createIdleSpeedState(),
    embedding: createIdleSpeedState(),
    rerank: createIdleSpeedState(),
    decision: createIdleSpeedState(),
  }
}

function createTestingSpeedTests(): Record<ConfigSection, SpeedTestState> {
  return {
    llm: createTestingSpeedState(),
    embedding: createTestingSpeedState(),
    rerank: createTestingSpeedState(),
    decision: createTestingSpeedState(),
  }
}

function speedSummary(state: SpeedTestState) {
  if (state.status === 'testing') return '测速中'
  if (state.status === 'error') return '测速失败'
  if (state.status === 'success' && state.latencyMs !== null) {
    return state.modelCount === null
      ? `${state.latencyMs} ms`
      : `${state.latencyMs} ms · ${state.modelCount} 个模型`
  }
  return '未测速'
}

function usageLabel(usage: ConfigSection): string {
  switch (usage) {
    case 'llm':
      return '大模型'
    case 'embedding':
      return 'Embedding'
    case 'rerank':
      return 'Rerank'
    case 'decision':
      return '决策模型'
  }
}

const sectionTitle: Record<ConfigSection, string> = {
  llm: '大模型配置',
  embedding: 'Embedding 配置',
  rerank: 'Rerank 配置',
  decision: '决策模型配置',
}

function decisionModeLabel(mode: DecisionMode): string {
  if (mode === 'shadow') return 'shadow 只记录'
  if (mode === 'enforce') return 'enforce 生效'
  return 'off 关闭'
}

function hostLabel(url: string) {
  try {
    return new URL(url).host
  } catch {
    return url || '未填写地址'
  }
}

function ModelCombobox({
  value,
  onChange,
  baseUrl,
  apiKey,
  section,
  placeholder,
}: {
  value: string
  onChange: (v: string) => void
  baseUrl: string
  apiKey: string
  section: ConfigSection
  placeholder?: string
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
    if (!baseUrl.trim()) {
      setError('请先填写 Base URL')
      return
    }
    setLoading(true)
    setError('')
    setModels([])
    setOpen(false)
    try {
      // 未填写新 key 时：非决策分区沿用测试接口；决策分区走 /models/fetch
      // 的 section 路径，由后端在本机、同 base URL 下使用已存密钥。
      const shouldFetchDirectly = Boolean(apiKey.trim()) || section === 'decision'
      const fetched = shouldFetchDirectly
        ? await fetchModels(baseUrl, apiKey, section)
        : (await testLlmConfig(section, baseUrl)).models.map(id => ({ id }))
      const filtered = filterModelsByUsage(fetched, section)
      setModels(filtered)
      setQuery('')
      setOpen(filtered.length > 0)
      const repairedValue = repairModelSelection(value, section, filtered)
      if (repairedValue && repairedValue !== value) {
        onChange(repairedValue)
      }
      if (fetched.length > 0 && filtered.length === 0) {
        setError(`未识别到可用于 ${usageLabel(section)} 的候选，请手动输入模型名`)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : '获取失败')
    } finally {
      setLoading(false)
    }
  }, [apiKey, baseUrl, onChange, section, value])

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
            <LoaderCircle className="h-3 w-3 animate-spin" />
          ) : (
            <RefreshCw className="h-3 w-3" />
          )}
          {loading ? '获取中...' : '获取模型'}
        </button>
      </div>
      <div className="relative" ref={containerRef}>
        <div className="flex items-center gap-0 rounded-sm border border-border bg-bg-body transition-[border-color,background-color] duration-150 ease-[var(--ease-out)] focus-within:border-primary-light focus-within:bg-white dark:border-white/10 dark:focus-within:bg-[#1e1f20]">
          <input
            className="h-10 flex-1 bg-transparent px-3 text-sm outline-none"
            id={`${section}-model`}
            name={`${section}-model`}
            placeholder={placeholder}
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
          <div className="absolute left-0 right-0 top-full z-10 mt-1 max-h-48 overflow-y-auto rounded-sm border border-border bg-white shadow-lg dark:border-white/10 dark:bg-[#1e1f20]">
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
          <div className="absolute left-0 right-0 top-full z-10 mt-1 rounded-sm border border-border bg-white px-3 py-2 text-xs text-text-muted shadow-lg dark:border-white/10 dark:bg-[#1e1f20]">
            无匹配模型
          </div>
        )}
      </div>
    </div>
  )
}

type SectionField = 'baseUrl' | 'apiKey' | 'model' | 'dimensions' | 'mode'

function ModelSectionEditor({
  draft,
  indexDimensions,
  onChange,
  snapshot,
  usage,
}: {
  draft: LlmConfigDraftSection & { dimensions?: number | null }
  indexDimensions: number | null
  onChange: (field: SectionField, value: string | number | null) => void
  snapshot: LlmConfigSection
  usage: ModelUsage
}) {
  const baseUrlPlaceholder = usage === 'rerank'
    ? 'https://api.example.com/v1'
    : 'https://api.openai.com/v1'

  return (
    <div className="grid gap-5">
      <label className="grid gap-1.5">
        <span className="text-xs font-semibold text-text-secondary">Base URL</span>
        <div className="flex items-center gap-2 rounded-sm border border-border bg-bg-body px-3 transition-[border-color,background-color] duration-150 ease-[var(--ease-out)] focus-within:border-primary-light focus-within:bg-white dark:border-white/10 dark:focus-within:bg-[#1e1f20]">
          <Server className="h-4 w-4 text-text-muted" />
          <input
            className="h-10 flex-1 bg-transparent text-sm outline-none"
            id={`${usage}-base-url`}
            name={`${usage}-base-url`}
            onChange={(event) => onChange('baseUrl', event.target.value)}
            placeholder={baseUrlPlaceholder}
            value={draft.baseUrl}
          />
        </div>
      </label>

      <label className="grid gap-1.5">
        <span className="text-xs font-semibold text-text-secondary">API Key</span>
        <div className="flex items-center gap-2 rounded-sm border border-border bg-bg-body px-3 transition-[border-color,background-color] duration-150 ease-[var(--ease-out)] focus-within:border-primary-light focus-within:bg-white dark:border-white/10 dark:focus-within:bg-[#1e1f20]">
          <Key className="h-4 w-4 text-text-muted" />
          <input
            className="h-10 flex-1 bg-transparent text-sm outline-none"
            id={`${usage}-api-key`}
            name={`${usage}-api-key`}
            onChange={(event) => onChange('apiKey', event.target.value)}
            placeholder={apiKeyPlaceholder(snapshot)}
            type="password"
            value={draft.apiKey}
          />
        </div>
      </label>

      <ModelCombobox
        apiKey={draft.apiKey}
        baseUrl={draft.baseUrl}
        onChange={(value) => onChange('model', value)}
        section={usage}
        value={draft.model}
      />

      {usage === 'embedding' && (
        <label className="grid gap-1.5">
          <span className="text-xs font-semibold text-text-secondary">向量维度</span>
          <input
            className="h-10 rounded-sm border border-border bg-bg-body px-3 text-sm outline-none transition-[border-color,background-color] duration-150 ease-[var(--ease-out)] focus:border-primary-light focus:bg-white dark:border-white/10 dark:focus:bg-[#1e1f20]"
            id="embedding-dimensions"
            max="4096"
            min="64"
            name="embedding-dimensions"
            onChange={(event) => onChange(
              'dimensions',
              event.target.value === '' ? null : Number(event.target.value),
            )}
            step="64"
            type="number"
            value={draft.dimensions ?? ''}
          />
          <span className="text-xs text-text-muted">当前索引维度：{indexDimensions ?? '未知'}</span>
        </label>
      )}
    </div>
  )
}

type DecisionPreset = 'siliconflow' | 'typesafe' | 'custom'

const DECISION_PRESETS: Record<
  Exclude<DecisionPreset, 'custom'>,
  { baseUrl: string; model: string }
> = {
  siliconflow: {
    baseUrl: 'https://api.siliconflow.cn/v1',
    model: 'Kev-4b',
  },
  typesafe: {
    baseUrl: 'https://api.typesafe.ai/v1',
    model: 'jev-latest',
  },
}

function decisionPresetFor(baseUrl: string): DecisionPreset {
  const normalized = baseUrl.trim().replace(/\/+$/, '')
  if (normalized === DECISION_PRESETS.siliconflow.baseUrl) return 'siliconflow'
  if (normalized === DECISION_PRESETS.typesafe.baseUrl) return 'typesafe'
  return 'custom'
}

function DecisionSectionEditor({
  draft,
  snapshot,
  onChange,
}: {
  draft: DecisionConfigDraft
  snapshot: LlmConfigDecisionSection
  onChange: (field: SectionField, value: string | number | null) => void
}) {
  const preset = decisionPresetFor(draft.baseUrl)

  return (
    <div className="grid gap-5">
      <label className="grid gap-1.5">
        <span className="text-xs font-semibold text-text-secondary">服务预设</span>
        <select
          className="h-10 rounded-sm border border-border bg-bg-body px-3 text-sm outline-none transition-[border-color,background-color] duration-150 ease-[var(--ease-out)] focus:border-primary-light focus:bg-white dark:border-white/10 dark:bg-[#1e1f20] dark:focus:bg-[#1e1f20]"
          id="decision-preset"
          name="decision-preset"
          onChange={(event) => {
            const next = event.target.value as DecisionPreset
            if (next === 'custom') return
            const target = DECISION_PRESETS[next]
            onChange('baseUrl', target.baseUrl)
            onChange('model', target.model)
            // Never forward a key typed for provider A to provider B.
            onChange('apiKey', '')
          }}
          value={preset}
        >
          <option value="siliconflow">硅基流动（参考）</option>
          <option value="typesafe">TypeSafe 官方</option>
          <option value="custom">自定义</option>
        </select>
      </label>

      <label className="grid gap-1.5">
        <span className="text-xs font-semibold text-text-secondary">Base URL</span>
        <div className="flex items-center gap-2 rounded-sm border border-border bg-bg-body px-3 transition-[border-color,background-color] duration-150 ease-[var(--ease-out)] focus-within:border-primary-light focus-within:bg-white dark:border-white/10 dark:focus-within:bg-[#1e1f20]">
          <Server className="h-4 w-4 text-text-muted" />
          <input
            className="h-10 flex-1 bg-transparent text-sm outline-none"
            id="decision-base-url"
            name="decision-base-url"
            onChange={(event) => onChange('baseUrl', event.target.value)}
            placeholder="https://api.siliconflow.cn/v1 或 https://api.typesafe.ai/v1"
            value={draft.baseUrl}
          />
        </div>
      </label>

      <label className="grid gap-1.5">
        <span className="text-xs font-semibold text-text-secondary">API Key</span>
        <div className="flex items-center gap-2 rounded-sm border border-border bg-bg-body px-3 transition-[border-color,background-color] duration-150 ease-[var(--ease-out)] focus-within:border-primary-light focus-within:bg-white dark:border-white/10 dark:focus-within:bg-[#1e1f20]">
          <Key className="h-4 w-4 text-text-muted" />
          <input
            className="h-10 flex-1 bg-transparent text-sm outline-none"
            id="decision-api-key"
            name="decision-api-key"
            onChange={(event) => onChange('apiKey', event.target.value)}
            placeholder={apiKeyPlaceholder(snapshot)}
            type="password"
            value={draft.apiKey}
          />
        </div>
      </label>

      <ModelCombobox
        apiKey={draft.apiKey}
        baseUrl={draft.baseUrl}
        onChange={(value) => onChange('model', value)}
        placeholder="获取模型或手动输入"
        section="decision"
        value={draft.model}
      />

      <label className="grid gap-1.5">
        <span className="text-xs font-semibold text-text-secondary">运行模式</span>
        <select
          className="h-10 rounded-sm border border-border bg-bg-body px-3 text-sm outline-none transition-[border-color,background-color] duration-150 ease-[var(--ease-out)] focus:border-primary-light focus:bg-white dark:border-white/10 dark:bg-[#1e1f20] dark:focus:bg-[#1e1f20]"
          id="decision-mode"
          name="decision-mode"
          onChange={(event) => onChange('mode', event.target.value)}
          value={draft.mode}
        >
          <option value="off">off — 不调用决策模型</option>
          <option value="shadow">shadow — 只记录，不改变门禁</option>
          <option value="enforce">enforce — 参与证据门禁</option>
        </select>
      </label>

      <p className="text-xs leading-relaxed text-text-muted">
        TypeSafe System One 标准协议：POST /systemone（Base URL 填 provider 的 /v1 根路径）。
        官方实现 https://api.typesafe.ai/v1 + jev-latest；硅基流动兼容实现
        https://api.siliconflow.cn/v1 + Kev-4b / SemIf。获取模型会读取 provider 的
        /models；没有该端点时可直接手输模型名。决策失败、超时或低于置信度阈值时回退关键词门禁。
      </p>
    </div>
  )
}

function LlmConfigDialog({ onClose }: { onClose: () => void }) {
  const { user } = useAuth()
  const [loadState, setLoadState] = useState<LoadState>('loading')
  const [loadError, setLoadError] = useState('')
  const [snapshot, setSnapshot] = useState<LlmConfigSnapshot | null>(null)
  const [draft, setDraft] = useState<LlmConfigDraft | null>(null)
  const [notice, setNotice] = useState('待保存')
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [saveFeedback, setSaveFeedback] = useState<SaveFeedback | null>(null)
  const [reindexWarning, setReindexWarning] = useState(false)
  const [speedTests, setSpeedTests] = useState<Record<ConfigSection, SpeedTestState>>(() => createIdleSpeedTests())
  const [editingUsage, setEditingUsage] = useState<ConfigSection | null>(null)



  const loadConfig = useCallback(async (successNotice = '已加载服务端配置') => {
    setLoadState('loading')
    setLoadError('')
    setSaveFeedback(null)
    setReindexWarning(false)
    setSpeedTests(createIdleSpeedTests())

    try {
      const next = await fetchLlmConfig()
      setSnapshot(next)
      setDraft(draftFromSnapshot(next))
      setNotice(successNotice)
      setLoadState('ready')
    } catch (error) {
      if (isLlmConfigError(error) && error.status === 404) {
        setSnapshot(null)
        setDraft(null)
        setLoadState('unsupported')
        return
      }
      setLoadError(error instanceof Error ? error.message : '加载配置失败')
      setLoadState('error')
    }
  }, [])

  useEffect(() => {
    void loadConfig()
  }, [loadConfig])

  const markDirty = () => {
    setNotice('待保存')
    setSaveFeedback(null)
    setReindexWarning(false)
    setSpeedTests(createIdleSpeedTests())
  }

  const updateLlm = <K extends keyof LlmConfigDraftSection>(field: K, value: LlmConfigDraftSection[K]) => {
    setDraft(prev => prev ? { ...prev, llm: { ...prev.llm, [field]: value } } : prev)
    markDirty()
  }

  const updateEmbedding = <K extends keyof LlmConfigDraft['embedding']>(field: K, value: LlmConfigDraft['embedding'][K]) => {
    setDraft(prev => prev ? { ...prev, embedding: { ...prev.embedding, [field]: value } } : prev)
    markDirty()
  }

  const updateRerank = <K extends keyof LlmConfigDraftSection>(field: K, value: LlmConfigDraftSection[K]) => {
    setDraft(prev => prev ? { ...prev, rerank: { ...prev.rerank, [field]: value } } : prev)
    markDirty()
  }

  const updateDecision = <K extends keyof DecisionConfigDraft>(field: K, value: DecisionConfigDraft[K]) => {
    setDraft(prev => prev ? { ...prev, decision: { ...prev.decision, [field]: value } } : prev)
    markDirty()
  }

  const updateEditingSection = (field: SectionField, value: string | number | null) => {
    if (!editingUsage) return
    const text = value === null ? '' : String(value)
    if (editingUsage === 'decision') {
      if (field === 'mode') updateDecision('mode', text as DecisionMode)
      else if (field === 'baseUrl') updateDecision('baseUrl', text)
      else if (field === 'apiKey') updateDecision('apiKey', text)
      else if (field === 'model') updateDecision('model', text)
      return
    }
    if (editingUsage === 'llm') {
      if (field === 'baseUrl') updateLlm('baseUrl', text)
      else if (field === 'apiKey') updateLlm('apiKey', text)
      else if (field === 'model') updateLlm('model', text)
      return
    }
    if (editingUsage === 'embedding') {
      if (field === 'dimensions') {
        updateEmbedding('dimensions', value === null ? null : Number(value))
      } else if (field === 'baseUrl') updateEmbedding('baseUrl', text)
      else if (field === 'apiKey') updateEmbedding('apiKey', text)
      else if (field === 'model') updateEmbedding('model', text)
      return
    }
    if (field === 'baseUrl') updateRerank('baseUrl', text)
    else if (field === 'apiKey') updateRerank('apiKey', text)
    else if (field === 'model') updateRerank('model', text)
  }

  const handleSave = async () => {
    if (!draft || loadState !== 'ready') return
    setSaving(true)
    setSaveFeedback(null)
    setReindexWarning(false)
    setNotice('保存中...')

    try {
      const result = await saveLlmConfig(draft)
      const { reindexRecommended, ...nextSnapshot } = result
      setSnapshot(nextSnapshot)
      setDraft(draftFromSnapshot(nextSnapshot))
      setSaveFeedback({ tone: 'success', title: '已保存并生效' })
      setReindexWarning(reindexRecommended)
      setNotice('已保存并生效')
    } catch (error) {
      const conflict = isLlmConfigError(error) && error.status === 409
      setSaveFeedback({
        tone: 'error',
        title: conflict ? '向量维度与当前索引不一致，保存被拒绝' : '保存失败',
        detail: error instanceof Error ? error.message : '保存失败',
        conflict,
      })
      setNotice(conflict ? '保存失败：维度冲突' : '保存失败')
    } finally {
      setSaving(false)
    }
  }

  const handleReset = () => {
    if (saving || testing) return
    void loadConfig('已还原为服务端配置')
  }

  const handleTestAll = async () => {
    if (!draft || loadState !== 'ready') return
    setTesting(true)
    setSpeedTests(createTestingSpeedTests())
    setNotice('测速中...')

    const testedAt = new Date().toISOString()
    const sections: ConfigSection[] = ['llm', 'embedding', 'rerank', 'decision']
    const results = await Promise.all(sections.map(async (section) => {
      const sectionDraft = draft[section]
      try {
        const measurement = await testLlmConfig(
          section,
          sectionDraft.baseUrl,
          sectionDraft.apiKey,
          section === 'decision' ? sectionDraft.model : undefined,
        )
        const message = section === 'decision'
          ? `决策结果：${measurement.models[0] ?? 'ok'}`
          : `返回 ${measurement.modelCount} 个模型`
        return [section, {
          status: 'success',
          latencyMs: measurement.latencyMs,
          modelCount: section === 'decision' ? 1 : measurement.modelCount,
          message,
          testedAt,
        } satisfies SpeedTestState] as const
      } catch (error) {
        return [section, {
          status: 'error',
          latencyMs: null,
          modelCount: null,
          message: error instanceof Error ? error.message : '测速失败',
          testedAt,
        } satisfies SpeedTestState] as const
      }
    }))

    const nextSpeedTests = Object.fromEntries(results) as Record<ConfigSection, SpeedTestState>
    setSpeedTests(nextSpeedTests)
    setTesting(false)

    if (Object.values(nextSpeedTests).every(item => item.status === 'error')) {
      setNotice('测速失败')
      return
    }
    if (Object.values(nextSpeedTests).some(item => item.status === 'error')) {
      setNotice('测速完成（含失败项）')
      return
    }
    setNotice('测速完成')
  }

  const actionsDisabled = loadState !== 'ready' || saving || testing

  const handleRestoreDefaults = async () => {
    if (actionsDisabled) return
    setSaving(true)
    setSaveFeedback(null)
    setReindexWarning(false)
    setNotice('正在清空账号配置...')
    try {
      const next = await resetLlmConfig()
      setSnapshot(next)
      setDraft(draftFromSnapshot(next))
      setSpeedTests(createIdleSpeedTests())
      setSaveFeedback({ tone: 'success', title: '已清空账号配置' })
      setNotice('已清空账号配置')
    } catch (error) {
      setSaveFeedback({
        tone: 'error',
        title: '清空账号配置失败',
        detail: error instanceof Error ? error.message : '清空失败',
      })
      setNotice('清空失败')
    } finally {
      setSaving(false)
    }
  }

  return (
    <>
      <ArcDialog
        actions={(
          <div className="flex items-center gap-2">
              <button
                className="inline-flex h-9 items-center gap-2 rounded-sm border border-border bg-white px-4 text-sm text-text-secondary transition-[transform,border-color,color,background-color] duration-150 ease-[var(--ease-out)] hover:border-primary-light hover:text-primary-light active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-50 dark:border-white/10 dark:bg-white/5"
                disabled={actionsDisabled}
                onClick={handleReset}
              >
                <RotateCcw className="h-4 w-4" />
                重置
              </button>
              <button
                className="inline-flex h-9 items-center gap-2 rounded-sm border border-border bg-white px-4 text-sm text-text-secondary transition-[transform,border-color,color,background-color] duration-150 ease-[var(--ease-out)] hover:border-primary-light hover:text-primary-light active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-50 dark:border-white/10 dark:bg-white/5"
                disabled={actionsDisabled}
                onClick={handleTestAll}
              >
                {testing ? (
                  <LoaderCircle className="h-4 w-4 animate-spin" />
                ) : (
                  <Gauge className="h-4 w-4" />
                )}
                {testing ? '测速中...' : '测速'}
              </button>
              <button
                className="inline-flex h-9 items-center gap-2 rounded-sm bg-primary px-4 text-sm text-white transition-[transform,background-color,opacity] duration-150 ease-[var(--ease-out)] hover:bg-primary-hover active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-70"
                disabled={actionsDisabled}
                onClick={handleSave}
              >
                {saving ? (
                  <LoaderCircle className="h-4 w-4 animate-spin" />
                ) : (
                  <Save className="h-4 w-4" />
                )}
                {saving ? '保存中...' : '保存'}
              </button>
            </div>
        )}
        description={snapshot ? `${notice} · 更新于 ${formatTimestamp(snapshot.updatedAt, '未保存')}` : notice}
        onClose={onClose}
        size="xl"
        title="模型配置"
      >
        <div className="flex flex-col gap-5">

        {snapshot && (
          <div className="flex flex-wrap items-center gap-x-3 gap-y-2 rounded-md border border-border bg-white px-3 py-2 text-xs text-text-secondary shadow-sm dark:border-white/10 dark:bg-[#1e1f20]">
            <span>
              当前账号 <b className="font-medium text-text-primary">{user?.login_id ?? ''}</b>
            </span>
            <span className="text-text-muted">配置只保存在当前账号；未填写的分项保持为空</span>
            <span className="flex flex-wrap items-center gap-1.5">
              {(['llm', 'embedding', 'rerank', 'decision'] as ConfigSection[]).map((section) => (
                <span
                  key={section}
                  className={`rounded-full border px-2 py-0.5 ${
                    snapshot.sources[section] === 'account'
                      ? 'border-[#1a3c2a]/20 bg-[#1a3c2a]/10 text-[#1a3c2a] dark:border-[#c2e7ff]/30 dark:bg-[#c2e7ff]/10 dark:text-[#c2e7ff]'
                      : 'border-border bg-black/[0.02] text-text-muted dark:border-white/10 dark:bg-white/5'
                  }`}
                >
                  {sectionTitle[section]} · {sourceLabel(snapshot.sources[section])}
                </span>
              ))}
            </span>
            <button
              type="button"
              onClick={() => void handleRestoreDefaults()}
              disabled={actionsDisabled}
              className="ml-auto inline-flex h-7 items-center gap-1.5 rounded-full border border-border bg-white px-3 text-[11px] text-text-secondary transition-colors hover:border-primary-light hover:text-primary-light disabled:cursor-not-allowed disabled:opacity-50 dark:border-white/10 dark:bg-white/5"
            >
              <Undo2 className="h-3.5 w-3.5" />
              清空配置
            </button>
          </div>
        )}

        {loadState === 'loading' && (
          <section className="flex items-center justify-center gap-3 rounded-md border border-border bg-white px-6 py-16 shadow-sm dark:border-white/10 dark:bg-[#1e1f20]">
            <LoaderCircle className="h-6 w-6 animate-spin text-primary-light" />
            <span className="text-sm text-text-secondary">正在加载服务端配置...</span>
          </section>
        )}

        {loadState === 'error' && (
          <section
            className="flex flex-col items-center gap-3 rounded-md border border-red-500/30 bg-red-500/5 px-6 py-12 text-center"
            role="alert"
          >
            <TriangleAlert className="h-8 w-8 text-red-500" />
            <div>
              <h2 className="text-base font-semibold text-text-primary">配置加载失败</h2>
              <p className="mt-1 break-words text-sm text-text-secondary">{loadError}</p>
            </div>
            <button
              className="mt-1 inline-flex h-9 items-center gap-2 rounded-sm border border-border bg-white px-4 text-sm text-text-secondary transition-[transform,border-color,color] duration-150 ease-[var(--ease-out)] hover:border-primary-light hover:text-primary-light active:scale-[0.97] dark:border-white/10 dark:bg-white/5"
              onClick={() => { void loadConfig() }}
            >
              <RefreshCw className="h-4 w-4" />
              重试
            </button>
          </section>
        )}

        {loadState === 'unsupported' && (
          <section
            className="flex flex-col items-center gap-3 rounded-md border border-border bg-white px-6 py-12 text-center shadow-sm dark:border-white/10 dark:bg-[#1e1f20]"
            role="status"
          >
            <Info className="h-8 w-8 text-text-muted" />
            <div>
              <h2 className="text-base font-semibold text-text-primary">当前部署不支持在线修改配置</h2>
              <p className="mt-1 text-sm text-text-secondary">
                该页面仅在本地开发环境可直连 ai-service 时可用，请修改 ai-service/.env 后重启服务。
              </p>
            </div>
            <button
              className="mt-1 inline-flex h-9 items-center gap-2 rounded-sm border border-border bg-white px-4 text-sm text-text-secondary transition-[transform,border-color,color] duration-150 ease-[var(--ease-out)] hover:border-primary-light hover:text-primary-light active:scale-[0.97] dark:border-white/10 dark:bg-white/5"
              onClick={() => { void loadConfig() }}
            >
              <RefreshCw className="h-4 w-4" />
              重新检测
            </button>
          </section>
        )}

        {loadState === 'ready' && snapshot && draft && (
          <>
            {saveFeedback && (
              <section
                className={`flex items-start gap-3 rounded-md border px-4 py-3 shadow-sm ${
                  saveFeedback.tone === 'success'
                    ? 'border-emerald-500/30 bg-emerald-500/10'
                    : saveFeedback.conflict
                      ? 'border-red-500/50 bg-red-500/10'
                      : 'border-red-500/30 bg-red-500/5'
                }`}
                role={saveFeedback.tone === 'error' ? 'alert' : 'status'}
              >
                {saveFeedback.tone === 'success' ? (
                  <CircleCheck className="mt-0.5 h-5 w-5 shrink-0 text-emerald-600 dark:text-emerald-400" />
                ) : (
                  <TriangleAlert className={`mt-0.5 h-5 w-5 shrink-0 ${saveFeedback.conflict ? 'text-red-600 dark:text-red-400' : 'text-red-500'}`} />
                )}
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className={`text-sm font-semibold ${saveFeedback.tone === 'success' ? 'text-emerald-700 dark:text-emerald-300' : 'text-red-600 dark:text-red-400'}`}>
                      {saveFeedback.title}
                    </span>
                    {saveFeedback.conflict && (
                      <span className="rounded-full bg-red-500/15 px-2 py-0.5 text-xs font-semibold text-red-600 dark:text-red-400">
                        409 维度冲突
                      </span>
                    )}
                  </div>
                  {saveFeedback.detail && (
                    <div className={`mt-1 break-words text-sm ${saveFeedback.tone === 'success' ? 'text-emerald-700/90 dark:text-emerald-300/90' : 'text-red-600/90 dark:text-red-400/90'}`}>
                      {saveFeedback.detail}
                    </div>
                  )}
                  {saveFeedback.conflict && (
                    <div className="mt-1 text-xs text-red-600/80 dark:text-red-400/80">
                      当前索引维度 {snapshot.indexDimensions ?? '未知'}，本次提交维度 {draft.embedding.dimensions ?? '未填写'}。请恢复原维度或重建索引后重试。
                    </div>
                  )}
                </div>
              </section>
            )}

            {reindexWarning && (
              <section
                className="flex items-start gap-3 rounded-md border border-amber-500/40 bg-amber-500/10 px-4 py-3 shadow-sm"
                role="status"
              >
                <TriangleAlert className="mt-0.5 h-5 w-5 shrink-0 text-amber-600 dark:text-amber-400" />
                <div>
                  <div className="text-sm font-semibold text-amber-700 dark:text-amber-300">建议重建向量索引</div>
                  <div className="mt-1 text-sm text-amber-700/90 dark:text-amber-300/90">
                    Embedding 的 Base URL 或模型已变化，下一轮检索会使用新配置；如向量维度也发生变化，请重建索引。
                  </div>
                </div>
              </section>
            )}

            <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              {([
                { usage: 'llm' as const, title: '大模型', icon: <Cpu className="h-5 w-5" /> },
                { usage: 'embedding' as const, title: 'Embedding', icon: <Database className="h-5 w-5" /> },
                { usage: 'rerank' as const, title: 'Rerank', icon: <Filter className="h-5 w-5" /> },
                { usage: 'decision' as const, title: '决策模型', icon: <Brain className="h-5 w-5" /> },
              ]).map(item => {
                const section = snapshot[item.usage]
                const keyStatus = section.apiKeySet ? '密钥已设置' : '密钥未设置'
                const meta = item.usage === 'embedding'
                  ? `维度 ${draft.embedding.dimensions ?? snapshot.indexDimensions ?? '未知'} · ${keyStatus}`
                  : `${hostLabel(section.baseUrl)} · ${keyStatus}`
                const status = item.usage === 'decision'
                  ? (speedTests.decision.status === 'idle'
                      ? decisionModeLabel(snapshot.decision.mode)
                      : speedTests.decision.message)
                  : speedSummary(speedTests[item.usage])
                return (
                  <ArcCard
                    action={<>配置<ChevronRight className="h-3.5 w-3.5" /></>}
                    description={section.model || '未配置'}
                    key={item.usage}
                    media={<span className="arc-card-icon">{item.icon}</span>}
                    meta={meta}
                    onOpen={() => setEditingUsage(item.usage)}
                    status={status}
                    title={item.title}
                  />
                )
              })}
            </section>
          </>
        )}
        </div>
      </ArcDialog>

      {editingUsage && snapshot && draft && (
        <ArcDialog
          description={snapshot[editingUsage].model || '未配置'}
          footer={(
            <div className="flex justify-end">
              <button
                className="inline-flex h-9 items-center gap-2 rounded-sm bg-primary px-4 text-sm text-white transition-colors duration-150 ease-[var(--ease-out)] hover:bg-primary-hover active:scale-[0.97]"
                onClick={() => setEditingUsage(null)}
                type="button"
              >
                完成
              </button>
            </div>
          )}
          initialFocus="first"
          onClose={() => setEditingUsage(null)}
          size="md"
          title={sectionTitle[editingUsage]}
        >
          {editingUsage === 'decision' ? (
            <DecisionSectionEditor
              draft={draft.decision}
              onChange={updateEditingSection}
              snapshot={snapshot.decision}
            />
          ) : (
            <ModelSectionEditor
              draft={editingUsage === 'embedding' ? draft.embedding : draft[editingUsage]}
              indexDimensions={snapshot.indexDimensions}
              onChange={updateEditingSection}
              snapshot={snapshot[editingUsage]}
              usage={editingUsage}
            />
          )}
        </ArcDialog>
      )}
    </>
  )
}

export default LlmConfigDialog
