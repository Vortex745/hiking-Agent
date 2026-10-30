import { useCallback, useEffect, useState } from 'react'
import { Download, LoaderCircle, Pencil, Trash2, TriangleAlert } from 'lucide-react'
import {
  clearV2Memories,
  deleteV2Memory,
  exportV2Memories,
  fetchV2Memories,
  fetchV2MemorySettings,
  updateV2Memory,
  updateV2MemorySettings,
  type V2Memory,
} from '../api/chatV2'
import { API } from '../api/config'
import { Button } from './arc/button/button'
import ArcDialog from './ui/dialog'

const PAGE_SIZE = 20

interface MemoryEditing {
  memoryId: string
  category: string
  content: string
  ttlDays: string
}

function formatMemoryTime(value: string | null) {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date)
}

export function MemoryDialog({ onClose, refreshKey = 0 }: { onClose: () => void; refreshKey?: number }) {
  const [enabled, setEnabled] = useState<boolean | null>(null)
  const [settingsPending, setSettingsPending] = useState(true)
  const [settingsKey, setSettingsKey] = useState(0)
  const [togglePending, setTogglePending] = useState(false)
  const [items, setItems] = useState<V2Memory[]>([])
  const [total, setTotal] = useState(0)
  const [offset, setOffset] = useState(0)
  const [listPending, setListPending] = useState(false)
  const [error, setError] = useState('')
  const [exporting, setExporting] = useState(false)
  const [confirmClear, setConfirmClear] = useState(false)
  const [clearPending, setClearPending] = useState(false)
  const [editing, setEditing] = useState<MemoryEditing | null>(null)
  const [editSaving, setEditSaving] = useState(false)
  const [editError, setEditError] = useState('')

  useEffect(() => {
    let cancelled = false
    setSettingsPending(true)
    setError('')
    void fetchV2MemorySettings()
      .then(result => { if (!cancelled) setEnabled(result.enabled) })
      .catch(err => {
        if (!cancelled) setError(err instanceof Error ? err.message : '记忆设置加载失败')
      })
      .finally(() => { if (!cancelled) setSettingsPending(false) })
    return () => { cancelled = true }
  }, [settingsKey])

  const reload = useCallback(async (nextOffset = offset) => {
    setListPending(true)
    setError('')
    try {
      const page = await fetchV2Memories(PAGE_SIZE, nextOffset)
      setItems(page.items)
      setTotal(page.total)
    } catch (err) {
      setError(err instanceof Error ? err.message : '记忆加载失败')
    } finally {
      setListPending(false)
    }
  }, [offset])

  useEffect(() => {
    if (enabled !== true) return
    void reload()
  }, [enabled, offset, refreshKey, reload])

  const handleToggle = async () => {
    if (enabled === null || togglePending) return
    setTogglePending(true)
    setError('')
    try {
      const result = await updateV2MemorySettings(!enabled)
      setEnabled(result.enabled)
      if (!result.enabled) {
        setItems([])
        setTotal(0)
        setOffset(0)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '记忆设置更新失败')
    } finally {
      setTogglePending(false)
    }
  }

  const handleExport = async () => {
    if (!enabled || exporting) return
    setExporting(true)
    setError('')
    try {
      const exported = await exportV2Memories()
      window.open(API.chatV2Artifact(exported.artifact_id), '_blank', 'noopener,noreferrer')
    } catch (err) {
      setError(err instanceof Error ? err.message : '记忆导出失败')
    } finally {
      setExporting(false)
    }
  }

  const handleClear = async () => {
    if (clearPending) return
    setClearPending(true)
    setError('')
    try {
      await clearV2Memories()
      setItems([])
      setTotal(0)
      setOffset(0)
      setConfirmClear(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : '记忆清除失败')
    } finally {
      setClearPending(false)
    }
  }

  const handleDelete = async (memoryId: string) => {
    setError('')
    try {
      await deleteV2Memory(memoryId)
      if (items.length === 1 && offset > 0) {
        setOffset(Math.max(0, offset - PAGE_SIZE))
      } else {
        await reload()
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '记忆删除失败')
    }
  }

  const startEdit = (item: V2Memory) => {
    setEditError('')
    setEditing({
      memoryId: item.memory_id,
      category: item.category,
      content: item.content,
      ttlDays: '',
    })
  }

  const handleSaveEdit = async () => {
    if (!editing || editSaving) return
    const ttlText = editing.ttlDays.trim()
    const ttlDays = ttlText ? Number(ttlText) : null
    if (!editing.content.trim()) {
      setEditError('记忆内容不能为空')
      return
    }
    if (!editing.category.trim()) {
      setEditError('类别不能为空')
      return
    }
    if (ttlDays !== null && (!Number.isInteger(ttlDays) || ttlDays < 1 || ttlDays > 3650)) {
      setEditError('有效期必须是 1 到 3650 天的整数')
      return
    }
    setEditSaving(true)
    setEditError('')
    try {
      await updateV2Memory(editing.memoryId, {
        category: editing.category.trim(),
        content: editing.content.trim(),
        ttl_days: ttlDays,
      })
      setEditing(null)
      await reload()
    } catch (err) {
      setEditError(err instanceof Error ? err.message : '记忆更新失败')
    } finally {
      setEditSaving(false)
    }
  }

  return (
    <ArcDialog
      actions={(
        <>
          <Button
            aria-label="导出长期记忆"
            disabled={!enabled || exporting}
            onClick={() => { void handleExport() }}
            title="导出 JSON"
            type="button"
            variant="ghost"
          >
            {exporting ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
          </Button>
          <Button
            aria-label="清除全部长期记忆"
            disabled={!enabled || total === 0 || clearPending}
            onClick={() => setConfirmClear(true)}
            title="清除全部"
            type="button"
            variant="ghost"
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </>
      )}
      onClose={onClose}
      size="md"
      title="长期记忆"
    >
      {settingsPending && enabled === null ? (
        <div className="flex items-center gap-3 text-sm text-text-secondary">
          <LoaderCircle className="h-4 w-4 animate-spin text-primary-light" />
          正在加载记忆设置...
        </div>
      ) : enabled === null ? (
        <div className="flex flex-col items-start gap-3 rounded-md border border-red-500/30 bg-red-500/5 px-4 py-3">
          <div className="flex items-center gap-2 text-sm text-red-600 dark:text-red-400">
            <TriangleAlert className="h-4 w-4 shrink-0" />
            {error || '记忆设置加载失败'}
          </div>
          <button
            className="inline-flex h-8 items-center rounded-sm border border-border bg-white px-3 text-xs text-text-secondary dark:border-white/10 dark:bg-white/5"
            onClick={() => setSettingsKey(key => key + 1)}
            type="button"
          >
            重试
          </button>
        </div>
      ) : (
        <>
          <div className="flex items-center justify-between gap-4 rounded-md border border-border bg-bg-card px-4 py-3 dark:border-white/10">
            <div className="min-w-0">
              <div className="text-sm font-medium text-text-primary">长期记忆</div>
              <div className="mt-0.5 text-xs text-text-muted">
                {enabled ? '已开启，对话会读取并更新记忆' : '已关闭，不会读取或写入记忆'}
              </div>
            </div>
            <button
              aria-checked={enabled}
              aria-label="启用长期记忆"
              className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors duration-150 ease-[var(--ease-out)] disabled:opacity-50 ${enabled ? 'bg-emerald-500' : 'bg-black/20 dark:bg-white/20'}`}
              disabled={togglePending}
              onClick={() => { void handleToggle() }}
              role="switch"
              type="button"
            >
              <span
                className={`absolute left-0.5 h-4 w-4 rounded-full bg-white shadow-sm transition-transform duration-150 ease-[var(--ease-out)] ${enabled ? 'translate-x-4' : 'translate-x-0'}`}
              />
            </button>
          </div>

          {error && (
            <div className="mt-4 flex items-center gap-2 rounded-md border border-red-500/30 bg-red-500/5 px-4 py-3 text-sm text-red-600 dark:text-red-400">
              <TriangleAlert className="h-4 w-4 shrink-0" />
              <span className="break-words">{error}</span>
            </div>
          )}

          {confirmClear && (
            <div className="mt-4 flex flex-wrap items-center justify-between gap-3 rounded-md border border-red-500/30 bg-red-500/5 px-4 py-3">
              <span className="text-sm text-red-600 dark:text-red-400">确认清除全部长期记忆？此操作无法撤销。</span>
              <div className="flex items-center gap-2">
                <button
                  className="inline-flex h-8 items-center rounded-sm border border-border bg-white px-3 text-xs text-text-secondary dark:border-white/10 dark:bg-white/5"
                  onClick={() => setConfirmClear(false)}
                  type="button"
                >
                  取消
                </button>
                <button
                  className="inline-flex h-8 items-center rounded-sm bg-red-600 px-3 text-xs text-white disabled:opacity-60"
                  disabled={clearPending}
                  onClick={() => { void handleClear() }}
                  type="button"
                >
                  {clearPending ? '清除中...' : '确认清除'}
                </button>
              </div>
            </div>
          )}

          {enabled && (
            <>
              <div className="mt-5 flex items-center justify-between text-xs text-text-muted">
                <span>共 {total} 条</span>
                {listPending && <LoaderCircle className="h-3.5 w-3.5 animate-spin" />}
              </div>

              <div className="mt-3 grid gap-2.5">
                {!listPending && items.length === 0 && (
                  <div className="rounded-md border border-dashed border-border px-4 py-10 text-center text-sm text-text-muted dark:border-white/10">
                    暂无长期记忆
                  </div>
                )}
                {items.map(item => (
                  <div
                    className="rounded-md border border-border bg-bg-card px-4 py-3 dark:border-white/10"
                    key={item.memory_id}
                  >
                    {editing?.memoryId === item.memory_id ? (
                      <div className="grid gap-3">
                        <label className="grid gap-1">
                          <span className="text-xs font-semibold text-text-secondary">类别</span>
                          <input
                            className="h-9 rounded-sm border border-border bg-bg-body px-3 text-sm outline-none focus:border-primary-light dark:border-white/10"
                            onChange={(event) => setEditing(current => current ? { ...current, category: event.target.value } : current)}
                            value={editing.category}
                          />
                        </label>
                        <label className="grid gap-1">
                          <span className="text-xs font-semibold text-text-secondary">内容</span>
                          <textarea
                            className="min-h-[84px] resize-y rounded-sm border border-border bg-bg-body px-3 py-2 text-sm outline-none focus:border-primary-light dark:border-white/10"
                            onChange={(event) => setEditing(current => current ? { ...current, content: event.target.value } : current)}
                            value={editing.content}
                          />
                        </label>
                        <label className="grid gap-1">
                          <span className="text-xs font-semibold text-text-secondary">有效期（天，留空永久）</span>
                          <input
                            className="h-9 rounded-sm border border-border bg-bg-body px-3 text-sm outline-none focus:border-primary-light dark:border-white/10"
                            min="1"
                            max="3650"
                            onChange={(event) => setEditing(current => current ? { ...current, ttlDays: event.target.value } : current)}
                            type="number"
                            value={editing.ttlDays}
                          />
                        </label>
                        {editError && <div className="text-xs text-red-500">{editError}</div>}
                        <div className="flex justify-end gap-2">
                          <button
                            className="inline-flex h-8 items-center rounded-sm border border-border bg-white px-3 text-xs text-text-secondary dark:border-white/10 dark:bg-white/5"
                            onClick={() => { setEditing(null); setEditError('') }}
                            type="button"
                          >
                            取消
                          </button>
                          <button
                            className="inline-flex h-8 items-center rounded-sm bg-primary px-3 text-xs text-white disabled:opacity-60"
                            disabled={editSaving}
                            onClick={() => { void handleSaveEdit() }}
                            type="button"
                          >
                            {editSaving ? '保存中...' : '保存'}
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <div className="text-[11px] font-medium text-emerald-600 dark:text-emerald-400">{item.category}</div>
                          <div className="mt-1 break-words text-sm text-text-primary">{item.content}</div>
                          <div className="mt-1 text-[11px] text-text-muted">
                            {formatMemoryTime(item.updated_at)}
                            {item.expires_at ? ` · 到期 ${formatMemoryTime(item.expires_at)}` : ''}
                          </div>
                        </div>
                        <div className="flex shrink-0 gap-1">
                          <button
                            aria-label="编辑记忆"
                            className="flex h-7 w-7 items-center justify-center rounded-sm text-text-muted transition-colors hover:bg-primary/10 hover:text-primary-light"
                            onClick={() => startEdit(item)}
                            title="编辑"
                            type="button"
                          >
                            <Pencil className="h-3.5 w-3.5" />
                          </button>
                          <button
                            aria-label="删除记忆"
                            className="flex h-7 w-7 items-center justify-center rounded-sm text-red-500 transition-colors hover:bg-red-500/10 hover:text-red-600"
                            onClick={() => { void handleDelete(item.memory_id) }}
                            title="删除"
                            type="button"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {total > PAGE_SIZE && (
                <div className="mt-4 flex items-center justify-between border-t border-border pt-3 text-xs text-text-secondary dark:border-white/10">
                  <button
                    className="disabled:opacity-40"
                    disabled={offset === 0}
                    onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
                    type="button"
                  >
                    上一页
                  </button>
                  <span>{offset + 1}-{Math.min(offset + PAGE_SIZE, total)} / {total}</span>
                  <button
                    className="disabled:opacity-40"
                    disabled={offset + PAGE_SIZE >= total}
                    onClick={() => setOffset(offset + PAGE_SIZE)}
                    type="button"
                  >
                    下一页
                  </button>
                </div>
              )}
            </>
          )}
        </>
      )}
    </ArcDialog>
  )
}

export default MemoryDialog
