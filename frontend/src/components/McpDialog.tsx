import { useCallback, useEffect, useRef, useState } from 'react'
import { ArrowLeft, ClipboardPaste, LoaderCircle, Pencil, Plug, Plus, Trash2 } from 'lucide-react'
import ArcDialog from './ui/dialog'
import { Button } from './arc/button/button'
import { Input } from './arc/input/input'
import { RadioGroup } from './arc/radio-group/radio-group'
import { Switch } from './arc/switch/switch'
import { Textarea } from './arc/textarea/textarea'
import { clearBuiltinMcpCredentials, deleteMcpServer, getMcpOAuthStatus, listMcpServers, saveBuiltinMcpCredentials, saveMcpServer, testMcpServer, type McpDraft, type McpServer } from '../api/mcp'
import { importMcpChoice, readMcpImport, type McpImportChoice } from './mcpImport'
import './McpDialog.css'

type Form = {
  name: string
  transport: 'http' | 'stdio'
  url: string
  command: string
  args: string
  credentials: string
  confirmed: boolean
}

const emptyForm = (): Form => ({ name: '', transport: 'http', url: '', command: '', args: '[]', credentials: '', confirmed: false })
const statusLabel: Record<McpServer['status'], string> = {
  configured: '已配置', unconfigured: '未配置', untested: '未测试', success: '测试成功', failed: '测试失败',
}

const CREDENTIAL_LABELS: Record<string, string> = { AMAP_API_KEY: '高德 API Key' }
const credentialLabel = (name: string) => CREDENTIAL_LABELS[name] ?? name

function asDraft(form: Form, server: McpServer | null, testId?: string): McpDraft {
  const args: unknown = JSON.parse(form.args || '[]')
  if (!Array.isArray(args) || args.some(value => typeof value !== 'string')) throw new Error('参数必须是字符串数组')
  const credentials: unknown = form.credentials.trim() ? JSON.parse(form.credentials) : null
  if (credentials !== null && (typeof credentials !== 'object' || Array.isArray(credentials) || Object.values(credentials).some(value => typeof value !== 'string'))) {
    throw new Error('请求头或环境变量必须是字符串键值对象')
  }
  const changed = server && (server.transport !== form.transport || (server.url || '') !== form.url.trim()
    || (server.command || '') !== form.command.trim() || JSON.stringify(server.args || []) !== JSON.stringify(args)
    || form.credentials.trim() !== '')
  return {
    name: form.name.trim(), transport: form.transport, url: form.url.trim(), command: form.command.trim(), args,
    headers: form.transport === 'http' ? credentials as Record<string, string> | null : {},
    env: form.transport === 'stdio' ? credentials as Record<string, string> | null : {},
    enabled: Boolean(server?.enabled && (!changed || testId)), test_id: testId,
    server_id: server?.id, stdio_confirmed: form.confirmed,
  }
}

export function McpDialog({ onClose }: { onClose: () => void }) {
  const [servers, setServers] = useState<McpServer[]>([])
  const [loading, setLoading] = useState(true)
  const [available, setAvailable] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [editing, setEditing] = useState<McpServer | null | undefined>(undefined)
  const [form, setForm] = useState<Form>(emptyForm)
  const [importing, setImporting] = useState(false)
  const [importText, setImportText] = useState('')
  const [importChoices, setImportChoices] = useState<McpImportChoice[] | null>(null)
  const [selectedChoice, setSelectedChoice] = useState('')
  const nameRef = useRef<HTMLInputElement>(null)
  const importRef = useRef<HTMLTextAreaElement>(null)
  const [testResult, setTestResult] = useState<{ test_id: string; tools: string[] } | null>(null)
  const [testFailed, setTestFailed] = useState(false)
  const [authFlow, setAuthFlow] = useState<{ flow_id: string; auth_url: string } | null>(null)
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)
  const [builtinEditing, setBuiltinEditing] = useState<McpServer | null>(null)
  const [builtinValues, setBuiltinValues] = useState<Record<string, string>>({})

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      setError('')
      setServers((await listMcpServers()).servers)
      setAvailable(true)
    } catch (reason) {
      setAvailable(false)
      setServers([])
      setError(reason instanceof Error ? reason.message : '无法加载 MCP 服务')
    } finally {
      setLoading(false)
    }
  }, [])
  useEffect(() => { void refresh() }, [refresh])
  useEffect(() => {
    if (editing !== undefined) (importing ? importRef.current : nameRef.current)?.focus()
  }, [editing, importing])
  useEffect(() => {
    if (!authFlow) return
    let stopped = false
    let timer = 0
    const poll = async () => {
      try {
        const status = await getMcpOAuthStatus(authFlow.flow_id)
        if (stopped) return
        if (status.status === 'success' && status.test_id) {
          setTestResult({ test_id: status.test_id, tools: status.tools || [] })
          setTestFailed(false)
          setAuthFlow(null)
          setNotice('授权成功，测试通过。')
          return
        }
        if (status.status === 'error') {
          setAuthFlow(null)
          setTestFailed(true)
          setError(status.detail || '授权失败，请重试')
          return
        }
      } catch {
        // Transient failures while the callback completes: keep polling.
      }
      if (!stopped) timer = window.setTimeout(poll, 1200)
    }
    timer = window.setTimeout(poll, 800)
    return () => {
      stopped = true
      window.clearTimeout(timer)
    }
  }, [authFlow])

  const update = (patch: Partial<Form>) => {
    setForm(current => ({ ...current, ...patch }))
    setTestResult(null)
    setAuthFlow(null)
    setError('')
  }
  const edit = (server: McpServer | null) => {
    setEditing(server)
    setImporting(false)
    setImportText('')
    setImportChoices(null)
    setSelectedChoice('')
    setNotice('')
    setForm(server ? {
      name: server.name, transport: server.transport, url: server.url || '', command: server.command || '',
      args: JSON.stringify(server.args || []), credentials: '', confirmed: false,
    } : emptyForm())
    setTestResult(null)
    setTestFailed(false)
    setAuthFlow(null)
    setError('')
  }

  const importConfig = () => {
    setError('')
    try {
      const choices = importChoices || readMcpImport(importText)
      if (!importChoices && choices.length > 1) {
        setImportChoices(choices)
        setSelectedChoice('')
        return
      }
      const choice = importChoices ? choices.find(item => item.name === selectedChoice) : choices[0]
      if (!choice) throw new Error('请选择要导入的服务')
      setForm(importMcpChoice(choice))
      setTestResult(null)
      setTestFailed(false)
      setImportText('')
      setImportChoices(null)
      setSelectedChoice('')
      setImporting(false)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '无法导入配置')
    }
  }

  async function test() {
    setBusy(true)
    setError('')
    setNotice('')
    try {
      const draft = asDraft(form, editing || null)
      if (draft.transport === 'stdio' && !draft.stdio_confirmed) throw new Error('请先确认允许在本机启动命令')
      const result = await testMcpServer(draft)
      if (result.status === 'auth_required') {
        setTestResult(null)
        setTestFailed(false)
        setAuthFlow({ flow_id: result.flow_id, auth_url: result.auth_url })
        setNotice('该服务需要 OAuth 授权，完成授权后会自动继续。')
        window.open(result.auth_url, '_blank', 'noopener,noreferrer')
        return
      }
      setTestResult({ test_id: result.test_id, tools: result.tools })
      setTestFailed(false)
    } catch (reason) {
      setTestResult(null)
      setTestFailed(true)
      setError(reason instanceof Error ? reason.message : '测试失败')
    } finally { setBusy(false) }
  }

  async function save() {
    setBusy(true)
    setError('')
    try {
      const draft = asDraft(form, editing || null, testResult?.test_id)
      if (testFailed || (editing?.status === 'failed' && !testResult)) draft.enabled = false
      await saveMcpServer(draft, editing?.id)
      setEditing(undefined)
      await refresh()
      if (editing === null) setNotice(testResult ? '服务已保存，可在列表中启用。' : '服务已保存，测试通过后可启用。')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '保存失败')
    } finally { setBusy(false) }
  }

  async function toggle(server: McpServer, enabled: boolean) {
    setBusy(true)
    setError('')
    setNotice('')
    try {
      await saveMcpServer({ name: server.name, transport: server.transport, url: server.url || '',
        command: server.command || '', args: server.args || [], headers: null, env: null, enabled }, server.id)
      await refresh()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '操作失败')
    } finally { setBusy(false) }
  }

  async function remove(id: string) {
    setBusy(true)
    try { await deleteMcpServer(id); setConfirmDelete(null); await refresh() }
    catch (reason) { setError(reason instanceof Error ? reason.message : '删除失败') }
    finally { setBusy(false) }
  }

  function openBuiltin(server: McpServer) {
    setBuiltinEditing(server)
    setBuiltinValues({})
    setEditing(undefined)
    setNotice('')
    setError('')
  }

  async function saveBuiltin() {
    if (!builtinEditing) return
    const env: Record<string, string> = {}
    for (const name of builtinEditing.env_names || []) {
      env[name] = (builtinValues[name] || '').trim()
    }
    setBusy(true)
    setError('')
    setNotice('')
    try {
      const next = await saveBuiltinMcpCredentials(env)
      setServers(next.servers)
      setBuiltinEditing(null)
      setNotice('已保存内置服务 Key。')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '保存失败')
    } finally {
      setBusy(false)
    }
  }

  async function clearBuiltin() {
    if (!builtinEditing) return
    setBusy(true)
    setError('')
    setNotice('')
    try {
      const next = await clearBuiltinMcpCredentials()
      setServers(next.servers)
      setBuiltinEditing(null)
      setNotice('已清除内置服务 Key。')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '清除失败')
    } finally {
      setBusy(false)
    }
  }

  const dialogTitle = builtinEditing
    ? `配置 ${builtinEditing.name}`
    : editing === undefined
      ? 'MCP 接入'
      : importing
        ? '粘贴 MCP 配置'
        : editing
          ? '编辑 MCP 服务'
          : '添加 MCP 服务'

  return (
    <ArcDialog onClose={onClose} size="md" title={dialogTitle}>
        {builtinEditing ? (
          <div className="mcp-panel">
            <button className="mcp-back" onClick={() => { setBuiltinEditing(null); setError('') }}><ArrowLeft size={16} /> 返回服务列表</button>
            {(builtinEditing.env_names || []).map(name => (
              <Input key={name} label={credentialLabel(name)} type="password"
                value={builtinValues[name] ?? ''}
                onChange={event => setBuiltinValues(prev => ({ ...prev, [name]: event.target.value }))}
                placeholder={builtinEditing.status === 'configured' ? '已设置，输入新值可替换' : '请填写服务 Key'}
                autoComplete="off" />
            ))}
            <p className="mcp-note">
              内置服务：{builtinEditing.transport.toUpperCase()} · {builtinEditing.command} {(builtinEditing.args || []).join(' ')}
            </p>
            <p className="mcp-note">工具：{builtinEditing.tools.join('、') || '—'}</p>
            <p className="mcp-note">高德天气与高德地图共用同一个 Key；密钥只保存在当前账号的本机配置中。</p>
            {error && <p role="alert" className="mcp-error">{error}</p>}
            <div className="mcp-form-actions">
              {builtinEditing.status === 'configured' && (
                <Button variant="secondary" loading={busy} onClick={() => { void clearBuiltin() }}>清除 Key</Button>
              )}
              <Button loading={busy} disabled={busy || !(builtinEditing.env_names || []).every(name => (builtinValues[name] || '').trim())}
                onClick={() => { void saveBuiltin() }}>保存</Button>
            </div>
          </div>
        ) : editing === undefined ? (
          <div className="mcp-panel">
            <div className="mcp-toolbar">
              <span>{available ? `${servers.length} 个服务` : ''}</span>
              <Button size="sm" disabled={loading || !available || busy} onClick={() => edit(null)}><Plus size={15} /> 添加服务</Button>
            </div>
            {notice && <p role="status" className="mcp-success">{notice}</p>}
            {error && <p role="alert" className="mcp-error">{error}</p>}
            {!loading && !available && <Button size="sm" variant="secondary" onClick={() => { void refresh() }}>重试</Button>}
            {loading ? <div className="mcp-loading"><LoaderCircle size={18} className="animate-spin" /> 加载中</div> : (
              <div className="mcp-list">
                {servers.map(server => (
                  <div className="mcp-row" key={server.id}>
                    <Plug size={17} aria-hidden="true" />
                    <div className="mcp-row-main">
                      <div className="mcp-row-name">{server.name}</div>
                      <div className="mcp-row-meta">{server.transport.toUpperCase()} · {statusLabel[server.status]} · {server.tools.length} 个工具</div>
                    </div>
                    {server.builtin ? (
                      <div className="mcp-actions">
                        <span className="mcp-builtin">内置</span>
                        <button aria-label={`配置 ${server.name} 的 Key`} title="配置 Key" className="mcp-icon" onClick={() => openBuiltin(server)}><Pencil size={16} /></button>
                      </div>
                    ) : (
                      <div className="mcp-actions">
                        <Switch aria-label={`启用 ${server.name}`} checked={Boolean(server.enabled)} disabled={busy} onCheckedChange={value => { void toggle(server, value) }} />
                        <button aria-label={`编辑 ${server.name}`} title="编辑" className="mcp-icon" onClick={() => edit(server)}><Pencil size={16} /></button>
                        {confirmDelete === server.id ? (
                          <Button size="sm" variant="danger" onClick={() => { void remove(server.id) }}>确认删除</Button>
                        ) : (
                          <button aria-label={`删除 ${server.name}`} title="删除" className="mcp-icon" onClick={() => setConfirmDelete(server.id)}><Trash2 size={16} /></button>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : importing ? (
          <div className="mcp-panel">
            <button className="mcp-back" onClick={() => { setImporting(false); setError('') }}><ArrowLeft size={16} /> 返回服务配置</button>
            <Textarea ref={importRef} label="MCP 配置 JSON" rows={10} className="mcp-json" value={importText}
              onChange={event => { setImportText(event.target.value); setImportChoices(null); setSelectedChoice(''); setError('') }}
              placeholder={'{\n  "mcpServers": {\n    "example-server": {\n      "command": "npx",\n      "args": ["-y", "package-name"]\n    }\n  }\n}'}
              spellCheck={false} autoComplete="off" aria-invalid={Boolean(error)} aria-describedby={error ? 'mcp-import-error' : undefined} />
            {error && <p id="mcp-import-error" role="alert" className="mcp-error">{error}</p>}
            {importChoices && (
              <div className="mcp-import-options">
                <RadioGroup label="选择要导入的服务" value={selectedChoice} onValueChange={value => { setSelectedChoice(value); setError('') }}
                  options={importChoices.map(choice => ({ value: choice.name, label: choice.name }))} />
              </div>
            )}
            <p className="mcp-note">请确认配置来源，测试 stdio 服务时会在本机运行命令。</p>
            <div className="mcp-form-actions">
              <Button disabled={!importText.trim() || Boolean(importChoices && !selectedChoice)} onClick={importConfig}>
                {importChoices ? '导入所选服务' : '导入配置'}
              </Button>
            </div>
          </div>
        ) : (
          <div className="mcp-panel">
            <div className="mcp-form-toolbar">
              <button className="mcp-back" onClick={() => { setEditing(undefined); setAuthFlow(null); setTestResult(null); setError('') }}><ArrowLeft size={16} /> 返回服务列表</button>
              {!editing && <Button size="sm" variant="secondary" onClick={() => { setImporting(true); setError('') }}><ClipboardPaste size={15} /> 粘贴配置</Button>}
            </div>
            <Input ref={nameRef} label="名称" value={form.name} onChange={event => update({ name: event.target.value })} placeholder="例如：地图服务" />
            <RadioGroup label="连接方式" value={form.transport} onValueChange={value => update({ transport: value as Form['transport'], credentials: '' })}
              options={[{ value: 'http', label: 'HTTP' }, { value: 'stdio', label: 'stdio' }]} />
            {form.transport === 'http' ? (
              <Input label="服务 URL" value={form.url} onChange={event => update({ url: event.target.value })} placeholder="https://example.com/mcp" />
            ) : (
              <>
                <Input label="命令" value={form.command} onChange={event => update({ command: event.target.value })} placeholder="例如：npx" />
                <Input label="参数（JSON 数组）" value={form.args} onChange={event => update({ args: event.target.value })} placeholder={'["-y", "package-name"]'} />
              </>
            )}
            <Input label={form.transport === 'http' ? '请求头（JSON 对象）' : '环境变量（JSON 对象）'}
              type="password" value={form.credentials} onChange={event => update({ credentials: event.target.value })}
              placeholder={editing ? '留空保持已保存的值' : '可选'} autoComplete="off" />
            {form.transport === 'stdio' && (
              <label className="mcp-confirm">
                <input type="checkbox" checked={form.confirmed} onChange={event => update({ confirmed: event.target.checked })} />
                <span>允许在本机启动此命令进行测试</span>
              </label>
            )}
            {authFlow && (
              <div className="mcp-auth" role="status">
                <p className="mcp-note">正在等待 OAuth 授权。若浏览器拦截了自动打开，请点击下方按钮。</p>
                <Button variant="secondary" onClick={() => window.open(authFlow.auth_url, '_blank', 'noopener,noreferrer')}>
                  打开授权页
                </Button>
              </div>
            )}
            <p className="mcp-note">请求头和环境变量以明文保存在本机。</p>
            {error && <p role="alert" className="mcp-error">{error}</p>}
            {testResult && <p role="status" className="mcp-success">测试成功，发现 {testResult.tools.length} 个工具</p>}
            <div className="mcp-form-actions">
              <Button variant="secondary" loading={busy} disabled={Boolean(authFlow) || !form.name || (form.transport === 'http' ? !form.url : !form.command)} onClick={() => { void test() }}>测试连接</Button>
              <Button loading={busy} disabled={Boolean(authFlow) || !form.name || (form.transport === 'http' ? !form.url : !form.command)} onClick={() => { void save() }}>保存</Button>
            </div>
          </div>
        )}
    </ArcDialog>
  )
}
