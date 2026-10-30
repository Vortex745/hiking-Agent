import { Loader2, Mountain, RefreshCw } from 'lucide-react'
import { Routes, Route } from 'react-router-dom'
import { AuthProvider, useAuth } from './auth/AuthContext'
import LoginPage from './pages/LoginPage'
import SuperAgent from './pages/SuperAgent'

function Splash({ label }: { label: string }) {
  return (
    <div className="flex h-[100dvh] w-full flex-col items-center justify-center gap-3 bg-[#f0f4f9] text-[#5e6063] dark:bg-[#131314] dark:text-[#9aa0a6]">
      <span className="flex size-11 items-center justify-center rounded-2xl bg-[#1a3c2a] text-[#f0f4f9] dark:bg-[#c2e7ff] dark:text-[#001d35]">
        <Mountain className="size-5" aria-hidden="true" />
      </span>
      <div className="flex items-center gap-2 text-xs">
        <Loader2 className="size-3.5 animate-spin" aria-hidden="true" />
        {label}
      </div>
    </div>
  )
}

function OfflineScreen({
  message,
  onRetry,
}: {
  message: string
  onRetry: () => Promise<void>
}) {
  return (
    <div className="flex h-[100dvh] w-full items-center justify-center bg-[#f0f4f9] px-4 dark:bg-[#131314]">
      <div className="w-full max-w-sm rounded-2xl border border-black/5 bg-white p-6 text-center shadow-sm dark:border-white/10 dark:bg-[#1e1f20]">
        <p className="text-sm font-medium text-[#1f1f1f] dark:text-[#e3e3e3]">无法连接服务</p>
        <p className="mt-2 text-xs leading-5 text-[#5e6063] dark:text-[#9aa0a6]">{message}</p>
        <button
          type="button"
          onClick={() => void onRetry()}
          className="mt-5 inline-flex items-center gap-2 rounded-xl bg-[#1a3c2a] px-4 py-2 text-xs font-medium text-white transition-colors hover:bg-[#23543c] dark:bg-[#c2e7ff] dark:text-[#001d35] dark:hover:bg-[#b3ddf7]"
        >
          <RefreshCw className="size-3.5" aria-hidden="true" />
          重试
        </button>
      </div>
    </div>
  )
}

function AuthGate() {
  const { status, user, offlineMessage, retry } = useAuth()

  if (status === 'loading') return <Splash label="正在检查登录状态" />
  if (status === 'offline') return <OfflineScreen message={offlineMessage} onRetry={retry} />
  if (status === 'anon' || !user) return <LoginPage />

  return (
    <div className="relative flex h-[100dvh] w-full bg-bg-body overflow-hidden">
      <div className="flex-1 flex flex-col w-full h-full relative overflow-hidden">
        <main className="flex-1 overflow-y-auto flex flex-col relative w-full h-full">
          <Routes>
            <Route path="/" element={<SuperAgent />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}

function App() {
  return (
    <AuthProvider>
      <AuthGate />
    </AuthProvider>
  )
}

export default App
