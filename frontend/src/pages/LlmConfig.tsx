import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import AlertTriangle from 'reicon-react/icons/AlertTriangle'
import CheckCircle from 'reicon-react/icons/CheckCircle'
import CircleInfo from 'reicon-react/icons/CircleInfo'
import Cpu from 'reicon-react/icons/Cpu'
import ChevronDown from 'reicon-react/icons/ChevronDown'
import Database from 'reicon-react/icons/Database'
import Gauge from 'reicon-react/icons/Gauge'
import Key from 'reicon-react/icons/Key'
import Filter from 'reicon-react/icons/Filter'
import Loader from 'reicon-react/icons/Loader'
import Refresh from 'reicon-react/icons/Refresh'
import RotateLeft from 'reicon-react/icons/RotateLeft'
import Save from 'reicon-react/icons/Save'
import Server from 'reicon-react/icons/Server'
import Settings2 from 'reicon-react/icons/Settings2'
import Sliders from 'reicon-react/icons/Sliders'
import {
  draftFromSnapshot,
  fetchLlmConfig,
  fetchModels,
  filterModelList,
  filterModelsByUsage,
  isLlmConfigError,
  LlmConfigDraft,
  LlmConfigDraftSection,
  LlmConfigSection,
  LlmConfigSnapshot,
  LlmSection,
  ModelInfo,
  ModelUsage,
  repairModelSelection,
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
  if (!section.apiKeySet) return '未设置'
  if (!section.apiKeyMasked || section.apiKeyMasked === '已设置') return '已设置，留空保持不变'
  return `已设置（${section.apiKeyMasked}），留空保持不变`
}

function serverKeyLabel(section: LlmConfigSection): string {
  if (!section.apiKeySet) return '未设置'
  return section.apiKeyMasked || '已设置'
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
      return 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400'
    case 'error':
      return 'bg-red-500/10 text-red-500 dark:text-red-400'
    default:
      return 'bg-slate-100 text-text-muted dark:bg-white/5'
  }
}

function ModelCombobox({
  value,
  onChange,
  baseUrl,
  apiKey,
  section,
  usage,
}: {
  value: string
  onChange: (v: string) => void
  baseUrl: string
  apiKey: string
  section: LlmSection
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
    if (!baseUrl.trim()) {
      setError('请先填写 Base URL')
      return
    }
    setLoading(true)
    setError('')
    setModels([])
    setOpen(false)
    try {
      // 未填写新 key 时走后端测试接口：服务端用已存密钥取模型列表；
      // 若 Base URL 也被改了，后端会返回 400 并说明必须先提供 key。
      const fetched = apiKey.trim()
        ? await fetchModels(baseUrl, apiKey)
        : (await testLlmConfig(section, baseUrl)).models.map(id => ({ id }))
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
  }, [apiKey, baseUrl, onChange, section, usage, value])

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
            <Loader className="h-3 w-3 animate-spin" />
          ) : (
            <Refresh className="h-3 w-3" />
          )}
          {loading ? '获取中...' : '获取模型'}
        </button>
      </div>
      <div className="relative" ref={containerRef}>
        <div className="flex items-center gap-0 rounded-sm border border-border bg-bg-body transition-[border-color,background-color] duration-150 ease-[var(--ease-out)] focus-within:border-primary-light focus-within:bg-white dark:border-white/10 dark:focus-within:bg-[#1e1f20]">
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

function LlmConfig() {
  const [loadState, setLoadState] = useState<LoadState>('loading')
  const [loadError, setLoadError] = useState('')
  const [snapshot, setSnapshot] = useState<LlmConfigSnapshot | null>(null)
  const [draft, setDraft] = useState<LlmConfigDraft | null>(null)
  const [notice, setNotice] = useState('待保存')
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [saveFeedback, setSaveFeedback] = useState<SaveFeedback | null>(null)
  const [reindexWarning, setReindexWarning] = useState(false)
  const [speedTests, setSpeedTests] = useState<Record<ModelUsage, SpeedTestState>>(() => createIdleSpeedTests())

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

  const summary = useMemo(() => {
    if (!snapshot) return []
    return [
      { label: '大模型', value: snapshot.llm.model || '未配置' },
      { label: 'Embedding', value: snapshot.embedding.model || '未配置' },
      { label: 'Rerank', value: snapshot.rerank.model || '未配置' },
      { label: '密钥', value: serverKeyLabel(snapshot.llm) },
      { label: '更新时间', value: formatTimestamp(snapshot.updatedAt, '未保存') },
    ]
  }, [snapshot])

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
    const sections: ModelUsage[] = ['llm', 'embedding', 'rerank']
    const results = await Promise.all(sections.map(async (section) => {
      const sectionDraft = draft[section]
      try {
        const measurement = await testLlmConfig(section, sectionDraft.baseUrl, sectionDraft.apiKey)
        return [section, {
          status: 'success',
          latencyMs: measurement.latencyMs,
          modelCount: measurement.modelCount,
          message: `返回 ${measurement.modelCount} 个模型`,
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

    const nextSpeedTests = Object.fromEntries(results) as Record<ModelUsage, SpeedTestState>
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

  return (
    <div className="h-full overflow-y-auto bg-bg-body px-4 py-4 md:px-8 md:py-6">
      <div className="mx-auto flex w-full max-w-[1180px] flex-col gap-5">
        {/* Header */}
        <section className="t-panel-slide" data-open="true">
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-border pb-5 dark:border-white/10">
            <div className="flex items-center gap-3">
              <div className="t-icon-swap h-11 w-11 rounded-sm bg-primary text-white" data-state="a">
                <span className="t-icon flex h-11 w-11 items-center justify-center" data-icon="a">
                  <Settings2 className="h-5 w-5" />
                </span>
                <span className="t-icon flex h-11 w-11 items-center justify-center" data-icon="b">
                  <Sliders className="h-5 w-5" />
                </span>
              </div>
              <div>
                <h1 className="text-[24px] font-bold text-text-primary">LLM 配置</h1>
                <p className="text-sm text-text-secondary">大模型、Embedding 与 Rerank 参数</p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                className="inline-flex h-9 items-center gap-2 rounded-sm border border-border bg-white px-4 text-sm text-text-secondary transition-[transform,border-color,color,background-color] duration-150 ease-[var(--ease-out)] hover:border-primary-light hover:text-primary-light active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-50 dark:border-white/10 dark:bg-white/5"
                disabled={actionsDisabled}
                onClick={handleReset}
              >
                <RotateLeft className="h-4 w-4" />
                重置
              </button>
              <button
                className="inline-flex h-9 items-center gap-2 rounded-sm border border-border bg-white px-4 text-sm text-text-secondary transition-[transform,border-color,color,background-color] duration-150 ease-[var(--ease-out)] hover:border-primary-light hover:text-primary-light active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-50 dark:border-white/10 dark:bg-white/5"
                disabled={actionsDisabled}
                onClick={handleTestAll}
              >
                {testing ? (
                  <Loader className="h-4 w-4 animate-spin" />
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
                  <Loader className="h-4 w-4 animate-spin" />
                ) : (
                  <Save className="h-4 w-4" />
                )}
                {saving ? '保存中...' : '保存'}
              </button>
            </div>
          </div>
        </section>

        {loadState === 'loading' && (
          <section className="flex items-center justify-center gap-3 rounded-md border border-border bg-white px-6 py-16 shadow-sm dark:border-white/10 dark:bg-[#1e1f20]">
            <Loader className="h-6 w-6 animate-spin text-primary-light" />
            <span className="text-sm text-text-secondary">正在加载服务端配置...</span>
          </section>
        )}

        {loadState === 'error' && (
          <section
            className="flex flex-col items-center gap-3 rounded-md border border-red-500/30 bg-red-500/5 px-6 py-12 text-center"
            role="alert"
          >
            <AlertTriangle className="h-8 w-8 text-red-500" />
            <div>
              <h2 className="text-base font-semibold text-text-primary">配置加载失败</h2>
              <p className="mt-1 break-words text-sm text-text-secondary">{loadError}</p>
            </div>
            <button
              className="mt-1 inline-flex h-9 items-center gap-2 rounded-sm border border-border bg-white px-4 text-sm text-text-secondary transition-[transform,border-color,color] duration-150 ease-[var(--ease-out)] hover:border-primary-light hover:text-primary-light active:scale-[0.97] dark:border-white/10 dark:bg-white/5"
              onClick={() => { void loadConfig() }}
            >
              <Refresh className="h-4 w-4" />
              重试
            </button>
          </section>
        )}

        {loadState === 'unsupported' && (
          <section
            className="flex flex-col items-center gap-3 rounded-md border border-border bg-white px-6 py-12 text-center shadow-sm dark:border-white/10 dark:bg-[#1e1f20]"
            role="status"
          >
            <CircleInfo className="h-8 w-8 text-text-muted" />
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
              <Refresh className="h-4 w-4" />
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
                  <CheckCircle className="mt-0.5 h-5 w-5 shrink-0 text-emerald-600 dark:text-emerald-400" />
                ) : (
                  <AlertTriangle className={`mt-0.5 h-5 w-5 shrink-0 ${saveFeedback.conflict ? 'text-red-600 dark:text-red-400' : 'text-red-500'}`} />
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
                <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-600 dark:text-amber-400" />
                <div>
                  <div className="text-sm font-semibold text-amber-700 dark:text-amber-300">建议重建向量索引</div>
                  <div className="mt-1 text-sm text-amber-700/90 dark:text-amber-300/90">
                    Embedding 的 Base URL 或模型已变化，下一轮检索会使用新配置；如向量维度也发生变化，请重建索引。
                  </div>
                </div>
              </section>
            )}

            {/* Summary Cards */}
            <section className="grid grid-cols-5 gap-3 max-xl:grid-cols-3 max-lg:grid-cols-2 max-sm:grid-cols-1">
              {summary.map((item, index) => (
                <div
                  className="rounded-sm border border-border bg-white px-4 py-3 shadow-sm dark:border-white/10 dark:bg-[#1e1f20]"
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
                  className="rounded-sm border border-border bg-white px-4 py-4 shadow-sm dark:border-white/10 dark:bg-[#1e1f20]"
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
                          {formatTimestamp(item.state.testedAt, '尚未测速')}
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
              <div className="t-panel-slide rounded-md border border-border bg-white p-5 shadow-sm dark:border-white/10 dark:bg-[#1e1f20]" data-open="true">
                <div className="mb-5 flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-sm bg-primary/10 text-primary-light">
                    <Cpu className="h-5 w-5" />
                  </div>
                  <div>
                    <h2 className="text-base font-semibold text-text-primary">大模型配置</h2>
                    <div className="text-xs text-text-muted">{notice}</div>
                  </div>
                </div>

                <div className="grid gap-4">
                  <label className="grid gap-1.5">
                    <span className="text-xs font-semibold text-text-secondary">Base URL</span>
                    <div className="flex items-center gap-2 rounded-sm border border-border bg-bg-body px-3 transition-[border-color,background-color] duration-150 ease-[var(--ease-out)] focus-within:border-primary-light focus-within:bg-white dark:border-white/10 dark:focus-within:bg-[#1e1f20]">
                      <Server className="h-4 w-4 text-text-muted" />
                      <input
                        className="h-10 flex-1 bg-transparent text-sm outline-none"
                        id="llm-base-url"
                        name="llm-base-url"
                        placeholder="https://api.openai.com/v1"
                        value={draft.llm.baseUrl}
                        onChange={(event) => updateLlm('baseUrl', event.target.value)}
                      />
                    </div>
                  </label>

                  <label className="grid gap-1.5">
                    <span className="text-xs font-semibold text-text-secondary">API Key</span>
                    <div className="flex items-center gap-2 rounded-sm border border-border bg-bg-body px-3 transition-[border-color,background-color] duration-150 ease-[var(--ease-out)] focus-within:border-primary-light focus-within:bg-white dark:border-white/10 dark:focus-within:bg-[#1e1f20]">
                      <Key className="h-4 w-4 text-text-muted" />
                      <input
                        className="h-10 flex-1 bg-transparent text-sm outline-none"
                        id="llm-api-key"
                        name="llm-api-key"
                        type="password"
                        placeholder={apiKeyPlaceholder(snapshot.llm)}
                        value={draft.llm.apiKey}
                        onChange={(event) => updateLlm('apiKey', event.target.value)}
                      />
                    </div>
                  </label>

                  <ModelCombobox
                    apiKey={draft.llm.apiKey}
                    baseUrl={draft.llm.baseUrl}
                    section="llm"
                    usage="llm"
                    value={draft.llm.model}
                    onChange={(v) => updateLlm('model', v)}
                  />
                </div>
              </div>

              {/* Embedding Config */}
              <div className="t-panel-slide rounded-md border border-border bg-white p-5 shadow-sm dark:border-white/10 dark:bg-[#1e1f20]" data-open="true">
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
                    <div className="flex items-center gap-2 rounded-sm border border-border bg-bg-body px-3 transition-[border-color,background-color] duration-150 ease-[var(--ease-out)] focus-within:border-primary-light focus-within:bg-white dark:border-white/10 dark:focus-within:bg-[#1e1f20]">
                      <Server className="h-4 w-4 text-text-muted" />
                      <input
                        className="h-10 flex-1 bg-transparent text-sm outline-none"
                        id="embedding-base-url"
                        name="embedding-base-url"
                        placeholder="https://api.openai.com/v1"
                        value={draft.embedding.baseUrl}
                        onChange={(event) => updateEmbedding('baseUrl', event.target.value)}
                      />
                    </div>
                  </label>

                  <label className="grid gap-1.5">
                    <span className="text-xs font-semibold text-text-secondary">API Key</span>
                    <div className="flex items-center gap-2 rounded-sm border border-border bg-bg-body px-3 transition-[border-color,background-color] duration-150 ease-[var(--ease-out)] focus-within:border-primary-light focus-within:bg-white dark:border-white/10 dark:focus-within:bg-[#1e1f20]">
                      <Key className="h-4 w-4 text-text-muted" />
                      <input
                        className="h-10 flex-1 bg-transparent text-sm outline-none"
                        id="embedding-api-key"
                        name="embedding-api-key"
                        type="password"
                        placeholder={apiKeyPlaceholder(snapshot.embedding)}
                        value={draft.embedding.apiKey}
                        onChange={(event) => updateEmbedding('apiKey', event.target.value)}
                      />
                    </div>
                  </label>

                  <ModelCombobox
                    apiKey={draft.embedding.apiKey}
                    baseUrl={draft.embedding.baseUrl}
                    section="embedding"
                    usage="embedding"
                    value={draft.embedding.model}
                    onChange={(v) => updateEmbedding('model', v)}
                  />

                  <label className="grid gap-1.5">
                    <span className="text-xs font-semibold text-text-secondary">向量维度</span>
                    <input
                      className="h-10 rounded-sm border border-border bg-bg-body px-3 text-sm outline-none transition-[border-color,background-color] duration-150 ease-[var(--ease-out)] focus:border-primary-light focus:bg-white dark:border-white/10 dark:focus:bg-[#1e1f20]"
                      id="embedding-dimensions"
                      min="64"
                      max="4096"
                      name="embedding-dimensions"
                      step="64"
                      type="number"
                      value={draft.embedding.dimensions ?? ''}
                      onChange={(event) => updateEmbedding(
                        'dimensions',
                        event.target.value === '' ? null : Number(event.target.value),
                      )}
                    />
                    <span className="text-xs text-text-muted">当前索引维度：{snapshot.indexDimensions ?? '未知'}</span>
                  </label>
                </div>
              </div>

              {/* Rerank Config */}
              <div className="t-panel-slide rounded-md border border-border bg-white p-5 shadow-sm dark:border-white/10 dark:bg-[#1e1f20]" data-open="true">
                <div className="mb-5 flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-sm bg-primary/10 text-primary-light">
                    <Filter className="h-5 w-5" />
                  </div>
                  <div>
                    <h2 className="text-base font-semibold text-text-primary">Rerank 模型配置</h2>
                    <div className="text-xs text-text-muted">{notice}</div>
                  </div>
                </div>

                <div className="grid gap-4">
                  <label className="grid gap-1.5">
                    <span className="text-xs font-semibold text-text-secondary">Base URL</span>
                    <div className="flex items-center gap-2 rounded-sm border border-border bg-bg-body px-3 transition-[border-color,background-color] duration-150 ease-[var(--ease-out)] focus-within:border-primary-light focus-within:bg-white dark:border-white/10 dark:focus-within:bg-[#1e1f20]">
                      <Server className="h-4 w-4 text-text-muted" />
                      <input
                        className="h-10 flex-1 bg-transparent text-sm outline-none"
                        id="rerank-base-url"
                        name="rerank-base-url"
                        placeholder="https://api.example.com/v1"
                        value={draft.rerank.baseUrl}
                        onChange={(event) => updateRerank('baseUrl', event.target.value)}
                      />
                    </div>
                  </label>

                  <label className="grid gap-1.5">
                    <span className="text-xs font-semibold text-text-secondary">API Key</span>
                    <div className="flex items-center gap-2 rounded-sm border border-border bg-bg-body px-3 transition-[border-color,background-color] duration-150 ease-[var(--ease-out)] focus-within:border-primary-light focus-within:bg-white dark:border-white/10 dark:focus-within:bg-[#1e1f20]">
                      <Key className="h-4 w-4 text-text-muted" />
                      <input
                        className="h-10 flex-1 bg-transparent text-sm outline-none"
                        id="rerank-api-key"
                        name="rerank-api-key"
                        type="password"
                        placeholder={apiKeyPlaceholder(snapshot.rerank)}
                        value={draft.rerank.apiKey}
                        onChange={(event) => updateRerank('apiKey', event.target.value)}
                      />
                    </div>
                  </label>

                  <ModelCombobox
                    apiKey={draft.rerank.apiKey}
                    baseUrl={draft.rerank.baseUrl}
                    section="rerank"
                    usage="rerank"
                    value={draft.rerank.model}
                    onChange={(v) => updateRerank('model', v)}
                  />
                </div>
              </div>
            </section>
          </>
        )}
      </div>
    </div>
  )
}

export default LlmConfig
