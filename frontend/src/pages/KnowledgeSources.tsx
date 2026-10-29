import { useCallback, useEffect, useMemo, useState } from 'react'
import AlertTriangle from 'reicon-react/icons/AlertTriangle'
import CheckCircle from 'reicon-react/icons/CheckCircle'
import Database from 'reicon-react/icons/Database'
import Loader from 'reicon-react/icons/Loader'
import Refresh from 'reicon-react/icons/Refresh'
import Trash2 from 'reicon-react/icons/Trash2'
import X from 'reicon-react/icons/X'
import {
  createDatasource,
  deleteDatasource,
  fetchDatasourceSync,
  fetchDatasourceSyncs,
  fetchDatasources,
  fetchDatasourceTypes,
  fetchSavedAncestors,
  fetchSavedResources,
  previewResources,
  startDatasourceSync,
  updateDatasource,
  validateCredentials,
  validateSavedDatasource,
  type ConnectorType,
  type Datasource,
  type ResourceNode,
  type SyncLog,
} from '../api/datasources'

interface KnowledgeSourcesDialogProps {
  onClose: () => void
}

type Notice = { tone: 'success' | 'error' | 'info'; message: string }
type View = { kind: 'list' } | { kind: 'form'; editing: Datasource | null } | { kind: 'logs'; ds: Datasource }

const BUTTON =
  'inline-flex h-9 items-center justify-center gap-2 rounded-sm border border-border bg-white px-3 text-sm text-text-secondary transition-[transform,border-color,color] duration-150 ease-[var(--ease-out)] hover:border-primary-light hover:text-primary-light active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-50 dark:border-white/10 dark:bg-white/5'
const PRIMARY =
  'inline-flex h-9 items-center justify-center gap-2 rounded-sm bg-primary px-4 text-sm text-white transition-[transform,background-color,opacity] duration-150 ease-[var(--ease-out)] hover:bg-primary-hover active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-60'
const DANGER =
  'inline-flex h-9 items-center justify-center gap-2 rounded-sm bg-red-600 px-4 text-sm text-white transition-[transform,background-color,opacity] duration-150 ease-[var(--ease-out)] hover:bg-red-500 active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-60'
const INPUT =
  'h-9 w-full rounded-sm border border-border bg-white px-3 text-sm text-text-primary outline-none transition-colors focus:border-primary-light dark:border-white/10 dark:bg-white/5'
const LABEL = 'text-xs font-medium text-text-secondary'

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function formatTime(value: string | null | undefined): string {
  if (!value) return '—'
  try {
    return new Intl.DateTimeFormat('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    }).format(new Date(value))
  } catch {
    return value
  }
}

function statusLabel(ds: Datasource): string {
  if (ds.sync_status === 'running') return '同步中'
  if (ds.last_sync_status === 'success') return '已完成'
  if (ds.last_sync_status === 'partial') return '部分完成'
  if (ds.last_sync_status === 'failed') return '失败'
  return '未同步'
}

function statusTone(ds: Datasource): string {
  if (ds.sync_status === 'running') return 'bg-sky-500/15 text-sky-600 dark:text-sky-300'
  if (ds.last_sync_status === 'success') return 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-300'
  if (ds.last_sync_status === 'failed') return 'bg-red-500/15 text-red-600 dark:text-red-300'
  if (ds.last_sync_status === 'partial') return 'bg-amber-500/15 text-amber-600 dark:text-amber-300'
  return 'bg-black/5 text-text-muted dark:bg-white/10'
}

function countsSummary(log: SyncLog | null): string {
  if (!log) return '尚无同步记录'
  const c = log.counts
  return `发现 ${c.discovered} · 抓取 ${c.fetched} · 新增 ${c.created} · 更新 ${c.updated} · 跳过 ${c.skipped} · 失败 ${c.failed} · 删除 ${c.deleted}`
}

export default function KnowledgeSourcesDialog({ onClose }: KnowledgeSourcesDialogProps) {
  const [types, setTypes] = useState<ConnectorType[]>([])
  const [sources, setSources] = useState<Datasource[]>([])
  const [view, setView] = useState<View>({ kind: 'list' })
  const [loading, setLoading] = useState(true)
  const [notice, setNotice] = useState<Notice | null>(null)
  const [busyId, setBusyId] = useState<string | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<Datasource | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    try {
      const [nextTypes, nextSources] = await Promise.all([fetchDatasourceTypes(), fetchDatasources()])
      setTypes(nextTypes)
      setSources(nextSources)
    } catch (error) {
      setNotice({ tone: 'error', message: errorMessage(error) })
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !deleteTarget) onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose, deleteTarget])

  const typeLabels = useMemo(() => {
    const map: Record<string, string> = {}
    for (const type of types) map[type.type] = type.label
    return map
  }, [types])

  const handleSync = useCallback(
    async (ds: Datasource) => {
      setBusyId(ds.id)
      setNotice({ tone: 'info', message: `正在同步「${ds.name}」…` })
      try {
        const started = await startDatasourceSync(ds.id)
        let finished: SyncLog | null = null
        for (let attempt = 0; attempt < 180; attempt += 1) {
          const log = await fetchDatasourceSync(ds.id, started.sync_id)
          if (log.status !== 'running') {
            finished = log
            break
          }
          await new Promise((resolve) => window.setTimeout(resolve, 1000))
        }
        if (!finished) {
          setNotice({ tone: 'info', message: '同步仍在进行，请稍后刷新查看结果' })
        } else {
          const tone = finished.status === 'failed' ? 'error' : finished.status === 'partial' ? 'info' : 'success'
          setNotice({ tone, message: `同步完成：${countsSummary(finished)}` })
        }
        await load()
      } catch (error) {
        setNotice({ tone: 'error', message: errorMessage(error) })
      } finally {
        setBusyId(null)
      }
    },
    [load],
  )

  const handleDelete = useCallback(
    async (ds: Datasource, purgeDocs: boolean) => {
      setBusyId(ds.id)
      try {
        await deleteDatasource(ds.id, purgeDocs)
        setDeleteTarget(null)
        setNotice({ tone: 'success', message: purgeDocs ? '已删除知识源并清空文档' : '已删除知识源（保留文档）' })
        await load()
      } catch (error) {
        setNotice({ tone: 'error', message: errorMessage(error) })
      } finally {
        setBusyId(null)
      }
    },
    [load],
  )

  const heading =
    view.kind === 'list'
      ? '已接入的知识源'
      : view.kind === 'form'
        ? view.editing
          ? `编辑「${view.editing.name}」`
          : '新建知识源'
        : `同步记录「${view.ds.name}」`

  return (
    <div
      className="fixed inset-0 z-[60] flex items-center justify-center bg-black/45 p-3 backdrop-blur-sm"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="知识源"
        className="flex h-[88vh] w-full max-w-[1080px] flex-col overflow-hidden rounded-lg border border-border bg-bg-body shadow-2xl dark:border-white/10"
      >
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-5 py-4 dark:border-white/10">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-sm bg-primary text-white">
              <Database className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-text-primary">知识源</h2>
              <p className="text-xs text-text-secondary">{heading}</p>
            </div>
          </div>
          <button type="button" className={BUTTON} aria-label="关闭知识源" onClick={onClose}>
            <X className="h-4 w-4" />
            关闭
          </button>
        </header>

        {notice && (
          <div
            role={notice.tone === 'error' ? 'alert' : 'status'}
            className={`flex items-start gap-2 border-b px-5 py-3 text-sm ${
              notice.tone === 'error'
                ? 'border-red-500/20 bg-red-500/5 text-red-600 dark:text-red-300'
                : notice.tone === 'success'
                  ? 'border-emerald-500/20 bg-emerald-500/5 text-emerald-700 dark:text-emerald-300'
                  : 'border-sky-500/20 bg-sky-500/5 text-sky-700 dark:text-sky-300'
            }`}
          >
            {notice.tone === 'error' ? (
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            ) : (
              <CheckCircle className="mt-0.5 h-4 w-4 shrink-0" />
            )}
            <span className="min-w-0 break-words">{notice.message}</span>
          </div>
        )}

        <div className="flex-1 overflow-y-auto px-5 py-4">
          {loading ? (
            <div className="flex items-center justify-center gap-3 py-20 text-sm text-text-secondary">
              <Loader className="h-5 w-5 animate-spin" />
              正在加载知识源…
            </div>
          ) : view.kind === 'list' ? (
            <div className="flex flex-col gap-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="text-sm text-text-secondary">
                  共 {sources.length} 个知识源。同步后的文档会立即进入聊天检索。
                </div>
                <div className="flex items-center gap-2">
                  <button type="button" className={BUTTON} onClick={() => void load()}>
                    <Refresh className="h-4 w-4" />
                    刷新
                  </button>
                  <button
                    type="button"
                    className={PRIMARY}
                    onClick={() => setView({ kind: 'form', editing: null })}
                    disabled={types.length === 0}
                  >
                    新建知识源
                  </button>
                </div>
              </div>

              {types.length === 0 && (
                <div className="rounded-md border border-border bg-white px-4 py-6 text-sm text-text-secondary dark:border-white/10 dark:bg-white/5">
                  当前服务尚未注册任何平台连接器。连接器接入后会出现在这里。
                </div>
              )}

              {sources.length === 0 ? (
                <div className="rounded-md border border-dashed border-border bg-white px-6 py-16 text-center dark:border-white/10 dark:bg-white/5">
                  <Database className="mx-auto h-8 w-8 text-text-muted" />
                  <h3 className="mt-3 text-base font-semibold text-text-primary">还没有知识源</h3>
                  <p className="mt-1 text-sm text-text-secondary">
                    选择一个平台，填写凭据并挑选要持续同步的空间或文档。
                  </p>
                  <button
                    type="button"
                    className={`${PRIMARY} mt-4`}
                    onClick={() => setView({ kind: 'form', editing: null })}
                    disabled={types.length === 0}
                  >
                    新建知识源
                  </button>
                </div>
              ) : (
                <div className="grid gap-3 md:grid-cols-2">
                  {sources.map((ds) => (
                    <article
                      key={ds.id}
                      className="flex flex-col gap-3 rounded-md border border-border bg-white p-4 shadow-sm dark:border-white/10 dark:bg-[#1e1f20]"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <div className="flex flex-wrap items-center gap-2">
                            <h3 className="truncate text-base font-semibold text-text-primary">{ds.name}</h3>
                            <span className="rounded-full bg-primary/10 px-2 py-0.5 text-xs text-primary-light">
                              {typeLabels[ds.type] ?? ds.type}
                            </span>
                          </div>
                          <div className="mt-1 text-xs text-text-secondary">
                            {ds.doc_count} 篇文档 · 排期 {ds.schedule || '仅手动'} · 上次 {formatTime(ds.last_sync_at)}
                          </div>
                        </div>
                        <span className={`shrink-0 rounded-full px-2 py-0.5 text-xs font-medium ${statusTone(ds)}`}>
                          {statusLabel(ds)}
                        </span>
                      </div>

                      <div className="rounded-sm bg-black/[0.03] px-3 py-2 text-xs text-text-secondary dark:bg-white/5">
                        {countsSummary(ds.last_sync)}
                        {ds.last_sync?.error_code ? ` · ${ds.last_sync.error_code}` : ''}
                      </div>

                      <div className="flex flex-wrap items-center gap-2">
                        <button
                          type="button"
                          className={PRIMARY}
                          disabled={ds.sync_status === 'running' || busyId === ds.id}
                          onClick={() => void handleSync(ds)}
                        >
                          {busyId === ds.id || ds.sync_status === 'running' ? (
                            <Loader className="h-4 w-4 animate-spin" />
                          ) : (
                            <Refresh className="h-4 w-4" />
                          )}
                          同步
                        </button>
                        <button
                          type="button"
                          className={BUTTON}
                          onClick={() => setView({ kind: 'form', editing: ds })}
                        >
                          编辑
                        </button>
                        <button
                          type="button"
                          className={BUTTON}
                          onClick={() => setView({ kind: 'logs', ds })}
                        >
                          日志
                        </button>
                        <button
                          type="button"
                          className={BUTTON}
                          onClick={() => setDeleteTarget(ds)}
                          aria-label={`删除知识源 ${ds.name}`}
                        >
                          <Trash2 className="h-4 w-4" />
                          删除
                        </button>
                      </div>
                    </article>
                  ))}
                </div>
              )}
            </div>
          ) : view.kind === 'form' ? (
            <SourceForm
              types={types}
              editing={view.editing}
              onCancel={() => setView({ kind: 'list' })}
              onSaved={async () => {
                setView({ kind: 'list' })
                setNotice({ tone: 'success', message: '知识源已保存' })
                await load()
              }}
              onNotice={setNotice}
            />
          ) : (
            <LogsPanel ds={view.ds} onBack={() => setView({ kind: 'list' })} />
          )}
        </div>
      </div>

      {deleteTarget && (
        <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/50 p-4">
          <div
            role="dialog"
            aria-modal="true"
            aria-label="删除知识源"
            className="w-full max-w-md rounded-lg border border-border bg-bg-body p-5 shadow-2xl dark:border-white/10"
          >
            <h3 className="text-base font-semibold text-text-primary">删除「{deleteTarget.name}」？</h3>
            <p className="mt-2 text-sm text-text-secondary">
              默认只删除连接，已入库的文档保留。也可以同时清空这个来源入库的文档。
            </p>
            <div className="mt-4 flex flex-wrap justify-end gap-2">
              <button type="button" className={BUTTON} onClick={() => setDeleteTarget(null)}>
                取消
              </button>
              <button
                type="button"
                className={BUTTON}
                disabled={busyId === deleteTarget.id}
                onClick={() => void handleDelete(deleteTarget, false)}
              >
                只删除连接
              </button>
              <button
                type="button"
                className={DANGER}
                disabled={busyId === deleteTarget.id}
                onClick={() => void handleDelete(deleteTarget, true)}
              >
                同时清空文档
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

interface SourceFormProps {
  types: ConnectorType[]
  editing: Datasource | null
  onCancel: () => void
  onSaved: () => Promise<void> | void
  onNotice: (notice: Notice | null) => void
}

function SourceForm({ types, editing, onCancel, onSaved, onNotice }: SourceFormProps) {
  const [type, setType] = useState(editing?.type ?? types[0]?.type ?? '')
  const meta = types.find((item) => item.type === type) ?? null
  const [name, setName] = useState(editing?.name ?? '')
  const [credentials, setCredentials] = useState<Record<string, string>>({})
  const [changeCredentials, setChangeCredentials] = useState(!editing)
  const [settings, setSettings] = useState<Record<string, string>>(() =>
    Object.fromEntries(
      (meta?.setting_fields ?? []).map((field) => [
        field.key,
        editing?.settings?.[field.key] != null ? String(editing.settings[field.key]) : field.default,
      ]),
    ),
  )
  const [selected, setSelected] = useState<string[]>(editing?.selected_resources ?? [])
  const [schedule, setSchedule] = useState(editing?.schedule ?? '')
  const [syncDeletions, setSyncDeletions] = useState(editing?.sync_deletions ?? true)
  const [nodes, setNodes] = useState<Record<string, ResourceNode[]>>({})
  const [fullyLoaded, setFullyLoaded] = useState<Record<string, boolean>>({})
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})
  const [resourceQuery, setResourceQuery] = useState('')
  const [treeLoading, setTreeLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [testing, setTesting] = useState(false)
  const [selectedTitles, setSelectedTitles] = useState<Record<string, string>>({})

  useEffect(() => {
    setCredentials({})
    setNodes({})
    setFullyLoaded({})
    setExpanded({})
    setSelectedTitles({})
    if (!editing) setSelected([])
    setSettings(
      Object.fromEntries(
        (meta?.setting_fields ?? []).map((field) => [
          field.key,
          editing?.settings?.[field.key] != null ? String(editing.settings[field.key]) : field.default,
        ]),
      ),
    )
  }, [type, editing?.id, meta?.setting_fields])

  useEffect(() => {
    if (!editing || editing.selected_resources.length === 0) return
    let cancelled = false
    fetchSavedAncestors(editing.id, editing.selected_resources)
      .then((ancestors) => {
        if (cancelled) return
        setSelectedTitles(Object.fromEntries(ancestors.map((node) => [node.id, node.title])))
        if (ancestors.length === 0) return
        setNodes((current) => {
          const next = { ...current }
          for (const node of ancestors) {
            const parentKey = node.parent_id ?? '__root__'
            const siblings = next[parentKey] ?? []
            if (!siblings.some((item) => item.id === node.id)) {
              next[parentKey] = [...siblings, node]
            }
          }
          return next
        })
        setExpanded((current) => {
          const next = { ...current }
          for (const node of ancestors) next[node.id] = true
          return next
        })
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [editing?.id, editing?.selected_resources])

  const credentialsReady = (meta?.auth_fields ?? []).every((field) => {
    if (!field.required) return true
    if (editing?.credentials_configured[field.key]) return true
    return Boolean(credentials[field.key]?.trim())
  })

  const credentialsForSubmit = useMemo(() => {
    const result: Record<string, string> = {}
    for (const [key, value] of Object.entries(credentials)) {
      if (value.trim()) result[key] = value.trim()
    }
    return result
  }, [credentials])

  const toggleSelected = (id: string) => {
    setSelected((current) =>
      current.includes(id) ? current.filter((item) => item !== id) : [...current, id],
    )
  }

  const loadRoots = async () => {
    if (!meta) return
    if (!credentialsReady && (!editing || changeCredentials)) {
      onNotice({ tone: 'error', message: '请先填写必填凭据，再加载资源' })
      return
    }
    setTreeLoading(true)
    try {
      const roots =
        editing && !changeCredentials
          ? await fetchSavedResources(editing.id, null)
          : await previewResources(type, credentialsForSubmit, settings, null)
      setNodes((current) => ({ ...current, __root__: roots }))
      setFullyLoaded((current) => ({ ...current, __root__: true }))
      setSelectedTitles((current) => {
        const next = { ...current }
        for (const node of roots) next[node.id] = node.title
        return next
      })
    } catch (error) {
      onNotice({ tone: 'error', message: errorMessage(error) })
    } finally {
      setTreeLoading(false)
    }
  }

  const toggleExpand = async (node: ResourceNode) => {
    if (expanded[node.id]) {
      setExpanded((current) => ({ ...current, [node.id]: false }))
      return
    }
    setExpanded((current) => ({ ...current, [node.id]: true }))
    if (nodes[node.id] && fullyLoaded[node.id]) return
    setTreeLoading(true)
    try {
      const children =
        editing && !changeCredentials
          ? await fetchSavedResources(editing.id, node.id)
          : await previewResources(type, credentialsForSubmit, settings, node.id)
      setNodes((current) => {
        const existing = current[node.id] ?? []
        const ids = new Set(existing.map((item) => item.id))
        return {
          ...current,
          [node.id]: [...existing, ...children.filter((item) => !ids.has(item.id))],
        }
      })
      setFullyLoaded((current) => ({ ...current, [node.id]: true }))
      setSelectedTitles((current) => {
        const next = { ...current }
        for (const child of children) next[child.id] = child.title
        return next
      })
    } catch (error) {
      onNotice({ tone: 'error', message: errorMessage(error) })
    } finally {
      setTreeLoading(false)
    }
  }

  const handleTest = async () => {
    setTesting(true)
    onNotice(null)
    try {
      if (editing && !changeCredentials) {
        await validateSavedDatasource(editing.id)
      } else {
        if (!credentialsReady) throw new Error('请先填写必填凭据')
        await validateCredentials(type, credentialsForSubmit, settings)
      }
      onNotice({ tone: 'success', message: '连接成功' })
    } catch (error) {
      onNotice({ tone: 'error', message: errorMessage(error) })
    } finally {
      setTesting(false)
    }
  }

  const handleSave = async () => {
    if (!meta) return
    if (!name.trim()) {
      onNotice({ tone: 'error', message: '请填写知识源名称' })
      return
    }
    if (selected.length === 0) {
      onNotice({ tone: 'error', message: '请至少选择一个资源' })
      return
    }
    if (changeCredentials && !credentialsReady) {
      onNotice({ tone: 'error', message: '请填写必填凭据' })
      return
    }
    setSaving(true)
    onNotice(null)
    try {
      if (editing) {
        await updateDatasource(editing.id, {
          name: name.trim(),
          ...(changeCredentials ? { credentials: credentialsForSubmit } : {}),
          settings,
          selected_resources: selected,
          schedule: schedule.trim(),
          sync_deletions: syncDeletions,
        })
      } else {
        await createDatasource({
          type,
          name: name.trim(),
          credentials: credentialsForSubmit,
          settings,
          selected_resources: selected,
          schedule: schedule.trim(),
          sync_deletions: syncDeletions,
        })
      }
      await onSaved()
    } catch (error) {
      onNotice({ tone: 'error', message: errorMessage(error) })
    } finally {
      setSaving(false)
    }
  }

  if (!meta) {
    return (
      <div className="rounded-md border border-border bg-white px-4 py-8 text-sm text-text-secondary dark:border-white/10 dark:bg-white/5">
        当前没有可用的连接器类型。
      </div>
    )
  }

  const roots = nodes.__root__ ?? []

  return (
    <div className="mx-auto flex w-full max-w-[860px] flex-col gap-5">
      <div className="grid gap-4 md:grid-cols-2">
        <label className="flex flex-col gap-1.5">
          <span className={LABEL}>平台</span>
          <select
            className={INPUT}
            value={type}
            disabled={Boolean(editing)}
            onChange={(event) => setType(event.target.value)}
          >
            {types.map((item) => (
              <option key={item.type} value={item.type}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1.5">
          <span className={LABEL}>名称</span>
          <input
            className={INPUT}
            value={name}
            placeholder="例如：徒步路线资料"
            onChange={(event) => setName(event.target.value)}
          />
        </label>
      </div>

      <section className="rounded-md border border-border bg-white p-4 dark:border-white/10 dark:bg-[#1e1f20]">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-text-primary">凭据</h3>
          {editing && (
            <label className="flex items-center gap-2 text-xs text-text-secondary">
              <input
                type="checkbox"
                className="h-4 w-4 accent-emerald-500"
                checked={changeCredentials}
                onChange={(event) => setChangeCredentials(event.target.checked)}
              />
              修改凭据
            </label>
          )}
        </div>
        {editing && !changeCredentials ? (
          <p className="mt-2 text-xs text-text-secondary">
            凭据已保存，界面不会回显。勾选“修改凭据”可重新填写。
          </p>
        ) : (
          <div className="mt-3 grid gap-3 md:grid-cols-2">
            {meta.auth_fields.map((field) => (
              <label className="flex flex-col gap-1.5" key={field.key}>
                <span className={LABEL}>
                  {field.label}
                  {field.required ? ' *' : ''}
                </span>
                <input
                  className={INPUT}
                  type={field.secret ? 'password' : 'text'}
                  autoComplete="off"
                  value={credentials[field.key] ?? ''}
                  placeholder={
                    field.placeholder ||
                    (field.secret && editing?.credentials_configured[field.key]
                      ? '已配置，留空保持不变'
                      : '')
                  }
                  onChange={(event) =>
                    setCredentials((current) => ({ ...current, [field.key]: event.target.value }))
                  }
                />
                {field.help && <span className="text-xs text-text-muted">{field.help}</span>}
              </label>
            ))}
          </div>
        )}
        {meta.setting_fields.length > 0 && (
          <div className="mt-4 grid gap-3 md:grid-cols-2">
            {meta.setting_fields.map((field) => (
              <label className="flex flex-col gap-1.5" key={field.key}>
                <span className={LABEL}>{field.label}</span>
                <input
                  className={INPUT}
                  value={settings[field.key] ?? field.default}
                  placeholder={field.placeholder}
                  onChange={(event) =>
                    setSettings((current) => ({ ...current, [field.key]: event.target.value }))
                  }
                />
                {field.help && <span className="text-xs text-text-muted">{field.help}</span>}
              </label>
            ))}
          </div>
        )}
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <button type="button" className={BUTTON} disabled={testing} onClick={() => void handleTest()}>
            {testing ? <Loader className="h-4 w-4 animate-spin" /> : null}
            测试连接
          </button>
          <button type="button" className={BUTTON} disabled={treeLoading} onClick={() => void loadRoots()}>
            {treeLoading ? <Loader className="h-4 w-4 animate-spin" /> : null}
            加载资源
          </button>
        </div>
      </section>

      <section className="rounded-md border border-border bg-white p-4 dark:border-white/10 dark:bg-[#1e1f20]">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-text-primary">同步范围</h3>
          <span className="text-xs text-text-secondary">已选 {selected.length} 项</span>
        </div>
        {selected.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-2">
            {selected.map((id) => (
              <span
                key={id}
                className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-2.5 py-1 text-xs text-primary-light"
              >
                {selectedTitles[id] ?? id}
                <button
                  type="button"
                  aria-label={`移除 ${selectedTitles[id] ?? id}`}
                  className="text-primary-light/70 hover:text-primary-light"
                  onClick={() => toggleSelected(id)}
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        )}
        {roots.length > 0 && (
          <input
            className={`${INPUT} mt-3`}
            value={resourceQuery}
            placeholder="搜索资源名称"
            onChange={(event) => setResourceQuery(event.target.value)}
          />
        )}
        <div className="mt-3 max-h-60 overflow-y-auto rounded-sm border border-border p-2 dark:border-white/10">
          {roots.length === 0 ? (
            <div className="px-2 py-6 text-center text-xs text-text-secondary">
              点击“加载资源”获取可选空间、知识库或页面。
            </div>
          ) : (
            <div className="flex flex-col gap-1">
              {roots
                .filter((node) => matchesNode(node, resourceQuery, nodes))
                .map((node) => (
                  <TreeNode
                    key={node.id}
                    node={node}
                    depth={0}
                    selected={selected}
                    expanded={expanded}
                    childrenByParent={nodes}
                    forceOpen={Boolean(resourceQuery.trim())}
                    onToggle={toggleSelected}
                    onExpand={toggleExpand}
                  />
                ))}
            </div>
          )}
        </div>
        <p className="mt-2 text-xs text-text-muted">
          选择空间/知识库/页面/目录即可包含其后代；同步删除只对本轮完整列举且仍选中的范围生效。
          不支持的平台类型（飞书表格/思维笔记/幻灯片/上传文件、Notion 数据库）会跳过并在日志计数。
        </p>
      </section>

      <section className="grid gap-4 rounded-md border border-border bg-white p-4 md:grid-cols-2 dark:border-white/10 dark:bg-[#1e1f20]">
        <label className="flex flex-col gap-1.5">
          <span className={LABEL}>定时同步（cron）</span>
          <input
            className={INPUT}
            value={schedule}
            placeholder="留空仅手动，例如 */30 * * * *"
            onChange={(event) => setSchedule(event.target.value)}
          />
          <span className="text-xs text-text-muted">5 字段：分 时 日 月 周；非法表达式会被拒绝。</span>
        </label>
        <label className="flex items-center gap-2 self-start rounded-sm border border-border px-3 py-2 text-sm text-text-secondary dark:border-white/10">
          <input
            type="checkbox"
            className="h-4 w-4 accent-emerald-500"
            checked={syncDeletions}
            onChange={(event) => setSyncDeletions(event.target.checked)}
          />
          同步源端删除
        </label>
      </section>

      <div className="flex flex-wrap justify-end gap-2">
        <button type="button" className={BUTTON} onClick={onCancel} disabled={saving}>
          取消
        </button>
        <button type="button" className={PRIMARY} onClick={() => void handleSave()} disabled={saving}>
          {saving ? <Loader className="h-4 w-4 animate-spin" /> : null}
          {editing ? '保存修改' : '创建知识源'}
        </button>
      </div>
    </div>
  )
}

function matchesNode(
  node: ResourceNode,
  query: string,
  childrenByParent: Record<string, ResourceNode[]>,
): boolean {
  const needle = query.trim().toLowerCase()
  if (!needle) return true
  if (
    node.title.toLowerCase().includes(needle) ||
    node.id.toLowerCase().includes(needle)
  ) {
    return true
  }
  return (childrenByParent[node.id] ?? []).some((child) =>
    matchesNode(child, needle, childrenByParent),
  )
}

interface TreeNodeProps {
  node: ResourceNode
  depth: number
  selected: string[]
  expanded: Record<string, boolean>
  childrenByParent: Record<string, ResourceNode[]>
  forceOpen?: boolean
  onToggle: (id: string) => void
  onExpand: (node: ResourceNode) => void
}

function TreeNode({
  node,
  depth,
  selected,
  expanded,
  childrenByParent,
  forceOpen = false,
  onToggle,
  onExpand,
}: TreeNodeProps) {
  const children = childrenByParent[node.id]
  const isExpanded = Boolean(expanded[node.id]) || forceOpen
  return (
    <div>
      <div
        className="flex items-center gap-2 rounded-sm px-1 py-1 hover:bg-black/[0.03] dark:hover:bg-white/5"
        style={{ paddingLeft: `${depth * 16 + 4}px` }}
      >
        <button
          type="button"
          className="flex h-5 w-5 items-center justify-center text-xs text-text-muted"
          aria-label={isExpanded ? `收起 ${node.title}` : `展开 ${node.title}`}
          onClick={() => onExpand(node)}
        >
          {isExpanded ? '▾' : '▸'}
        </button>
        <input
          type="checkbox"
          className="h-4 w-4 accent-emerald-500"
          checked={selected.includes(node.id)}
          onChange={() => onToggle(node.id)}
          aria-label={`选择 ${node.title}`}
        />
        <span className="min-w-0 truncate text-sm text-text-primary">{node.title || node.id}</span>
        <span className="shrink-0 text-[10px] uppercase tracking-wide text-text-muted">{node.kind}</span>
      </div>
      {isExpanded && children && children.length > 0 && (
        <div>
          {children.map((child) => (
            <TreeNode
              key={child.id}
              node={child}
              depth={depth + 1}
              selected={selected}
              expanded={expanded}
              childrenByParent={childrenByParent}
              forceOpen={forceOpen}
              onToggle={onToggle}
              onExpand={onExpand}
            />
          ))}
        </div>
      )}
    </div>
  )
}

function LogsPanel({ ds, onBack }: { ds: Datasource; onBack: () => void }) {
  const [logs, setLogs] = useState<SyncLog[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setLogs(await fetchDatasourceSyncs(ds.id, 20))
    } catch (cause) {
      setError(errorMessage(cause))
    } finally {
      setLoading(false)
    }
  }, [ds.id])

  useEffect(() => {
    void load()
  }, [load])

  return (
    <div className="mx-auto flex w-full max-w-[860px] flex-col gap-4">
      <div className="flex items-center justify-between">
        <button type="button" className={BUTTON} onClick={onBack}>
          返回列表
        </button>
        <button type="button" className={BUTTON} onClick={() => void load()}>
          <Refresh className="h-4 w-4" />
          刷新
        </button>
      </div>
      {error && (
        <div role="alert" className="rounded-md border border-red-500/30 bg-red-500/5 px-4 py-3 text-sm text-red-600">
          {error}
        </div>
      )}
      {loading ? (
        <div className="flex items-center justify-center gap-3 py-16 text-sm text-text-secondary">
          <Loader className="h-5 w-5 animate-spin" />
          正在加载同步记录…
        </div>
      ) : logs.length === 0 ? (
        <div className="rounded-md border border-dashed border-border px-6 py-12 text-center text-sm text-text-secondary dark:border-white/10">
          还没有同步记录。回到列表点击“同步”即可开始。
        </div>
      ) : (
        logs.map((log) => (
          <article
            key={log.id}
            className="rounded-md border border-border bg-white p-4 dark:border-white/10 dark:bg-[#1e1f20]"
          >
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-text-primary">
                  {log.trigger === 'schedule' ? '定时同步' : '手动同步'}
                </span>
                <span
                  className={`rounded-full px-2 py-0.5 text-xs ${
                    log.status === 'success'
                      ? 'bg-emerald-500/15 text-emerald-600'
                      : log.status === 'failed'
                        ? 'bg-red-500/15 text-red-600'
                        : log.status === 'partial'
                          ? 'bg-amber-500/15 text-amber-600'
                          : 'bg-sky-500/15 text-sky-600'
                  }`}
                >
                  {log.status}
                </span>
              </div>
              <span className="text-xs text-text-muted">
                {formatTime(log.started_at)} → {formatTime(log.finished_at)}
              </span>
            </div>
            <div className="mt-2 text-xs text-text-secondary">{countsSummary(log)}</div>
            {log.error_code && (
              <div className="mt-1 text-xs text-amber-600">
                {log.error_code}
                {log.message ? `：${log.message}` : ''}
              </div>
            )}
            {log.failures.length > 0 && (
              <details className="mt-2">
                <summary className="cursor-pointer text-xs text-text-secondary">
                  失败样本 {log.failures.length} 条
                </summary>
                <ul className="mt-2 flex flex-col gap-1">
                  {log.failures.map((failure, index) => (
                    <li key={`${failure.external_id}-${index}`} className="rounded-sm bg-red-500/5 px-2 py-1 text-xs text-red-600">
                      {failure.title || failure.external_id || '未知条目'} · {failure.error_code}
                      {failure.message ? `：${failure.message}` : ''}
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </article>
        ))
      )}
    </div>
  )
}
