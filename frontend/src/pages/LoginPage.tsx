import { useState, type FormEvent } from 'react'
import { Eye, EyeOff, Loader2 } from 'lucide-react'
import { AuthApiError } from '../api/auth'
import { useAuth } from '../auth/AuthContext'

type Mode = 'login' | 'register'

const LOGIN_ID_RE = /^[A-Za-z0-9_]{2,6}$/

function validate(mode: Mode, loginId: string, password: string, confirm: string) {
  if (!LOGIN_ID_RE.test(loginId)) return '登录 ID 需为 2-6 位字母、数字或下划线'
  if (password.length < 8 || password.length > 15) return '密码需为 8-15 个字符'
  if (/\s/.test(password)) return '密码不能包含空白字符'
  if (mode === 'register' && password !== confirm) return '两次输入的密码不一致'
  return ''
}

export default function LoginPage() {
  const { signIn, signUp } = useAuth()
  const [mode, setMode] = useState<Mode>('login')
  const [loginId, setLoginId] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const switchMode = (next: Mode) => {
    if (next === mode) return
    setMode(next)
    setError('')
    setConfirm('')
    setPassword('')
  }

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    if (submitting) return
    const message = validate(mode, loginId, password, confirm)
    if (message) {
      setError(message)
      return
    }
    setSubmitting(true)
    setError('')
    try {
      if (mode === 'login') await signIn(loginId, password)
      else await signUp(loginId, password)
    } catch (exc) {
      setError(
        exc instanceof AuthApiError || exc instanceof Error
          ? exc.message
          : '请求失败，请稍后重试',
      )
    } finally {
      setSubmitting(false)
    }
  }

  const inputClass =
    'w-full rounded-xl border border-black/10 bg-[#f8fafb] px-3.5 py-2.5 text-sm text-[#1f1f1f] outline-none transition-[border-color,box-shadow] placeholder:text-[#8a8f98] focus:border-[#1a3c2a] focus:ring-2 focus:ring-[#1a3c2a]/15 dark:border-white/10 dark:bg-white/5 dark:text-[#e3e3e3] dark:placeholder:text-[#8e918f] dark:focus:border-[#c2e7ff] dark:focus:ring-[#c2e7ff]/20'

  return (
    <div className="relative flex min-h-[100dvh] items-center justify-center overflow-hidden bg-[#f0f4f9] px-4 py-10 dark:bg-[#131314]">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -left-24 -top-24 h-72 w-72 rounded-full bg-[#1a3c2a]/10 blur-3xl dark:bg-[#c2e7ff]/10"
      />
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -bottom-32 -right-20 h-80 w-80 rounded-full bg-[#d3e3fd]/60 blur-3xl dark:bg-[#004a77]/20"
      />

      <div className="relative w-full max-w-[420px]">
        <div className="flex flex-col items-center text-center">
          <h1 className="text-2xl font-bold text-[#1f1f1f] dark:text-[#e3e3e3]">AI Hiking</h1>
        </div>

        <div className="mt-7 rounded-[28px] border border-black/5 bg-white/95 p-6 shadow-[0_24px_60px_-30px_rgba(15,40,25,0.45)] backdrop-blur dark:border-white/10 dark:bg-[#1e1f20]/95">
          <div
            role="tablist"
            aria-label="登录或注册"
            className="relative flex rounded-xl bg-black/[0.04] p-1 dark:bg-white/[0.06]"
          >
            <span
              aria-hidden="true"
              className={`pointer-events-none absolute inset-y-1 left-1 w-[calc(50%-4px)] rounded-lg bg-white shadow-sm transition-transform duration-200 ease-[var(--ease-out)] dark:bg-white/[0.12] ${
                mode === 'register' ? 'translate-x-full' : ''
              }`}
            />
            {(['login', 'register'] as const).map((item) => (
              <button
                key={item}
                type="button"
                role="tab"
                aria-selected={mode === item}
                onClick={() => switchMode(item)}
                className={`relative flex-1 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                  mode === item
                    ? 'text-[#1f1f1f] dark:text-[#e3e3e3]'
                    : 'text-[#5e6063] hover:text-[#1f1f1f] dark:text-[#9aa0a6] dark:hover:text-[#e3e3e3]'
                }`}
              >
                {item === 'login' ? '登录' : '注册'}
              </button>
            ))}
          </div>

          <form className="mt-5 space-y-4" onSubmit={submit} noValidate>
            <div>
              <label
                htmlFor="login-id"
                className="mb-1.5 block text-xs font-medium text-[#1f1f1f] dark:text-[#e3e3e3]"
              >
                登录 ID
              </label>
              <input
                id="login-id"
                name="loginId"
                autoComplete="username"
                autoCapitalize="none"
                spellCheck={false}
                maxLength={6}
                value={loginId}
                onChange={(event) => setLoginId(event.target.value)}
                placeholder="2-6 位字母或数字"
                className={inputClass}
              />
            </div>

            <div>
              <label
                htmlFor="login-password"
                className="mb-1.5 block text-xs font-medium text-[#1f1f1f] dark:text-[#e3e3e3]"
              >
                密码
              </label>
              <div className="relative">
                <input
                  id="login-password"
                  name="password"
                  type={showPassword ? 'text' : 'password'}
                  autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                  maxLength={15}
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  placeholder="8-15 个字符"
                  className={`${inputClass} pr-11`}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((value) => !value)}
                  aria-label={showPassword ? '隐藏密码' : '显示密码'}
                  className="absolute right-1.5 top-1/2 flex size-8 -translate-y-1/2 items-center justify-center rounded-lg text-[#5e6063] transition-[color,background-color,transform] duration-150 ease-[var(--ease-out)] hover:bg-black/5 active:scale-[0.97] dark:text-[#9aa0a6] dark:hover:bg-white/10"
                >
                  <span
                    className="t-icon-swap"
                    data-state={showPassword ? 'b' : 'a'}
                    aria-hidden="true"
                  >
                    <Eye className="t-icon size-4" data-icon="a" />
                    <EyeOff className="t-icon size-4" data-icon="b" />
                  </span>
                </button>
              </div>
            </div>

            {mode === 'register' && (
              <div>
                <label
                  htmlFor="login-confirm"
                  className="mb-1.5 block text-xs font-medium text-[#1f1f1f] dark:text-[#e3e3e3]"
                >
                  确认密码
                </label>
                <input
                  id="login-confirm"
                  name="confirm"
                  type={showPassword ? 'text' : 'password'}
                  autoComplete="new-password"
                  maxLength={15}
                  value={confirm}
                  onChange={(event) => setConfirm(event.target.value)}
                  placeholder="再次输入密码"
                  className={inputClass}
                />
              </div>
            )}

            {error && (
              <p
                role="alert"
                className="alert-in rounded-xl bg-[#fce8e6] px-3 py-2 text-xs leading-5 text-[#b3261e] dark:bg-[#5c1d18]/60 dark:text-[#f2b8b5]"
              >
                {error}
              </p>
            )}

            <button
              type="submit"
              disabled={submitting}
              className="flex w-full items-center justify-center gap-2 rounded-xl bg-[#1a3c2a] px-4 py-2.5 text-sm font-medium text-white transition-[transform,background-color] duration-150 ease-[var(--ease-out)] hover:bg-[#23543c] active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-60 dark:bg-[#c2e7ff] dark:text-[#001d35] dark:hover:bg-[#b3ddf7]"
            >
              {submitting && <Loader2 className="size-4 animate-spin" aria-hidden="true" />}
              {mode === 'login' ? '登录' : '创建账号'}
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}
