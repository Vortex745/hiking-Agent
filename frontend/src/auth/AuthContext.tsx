import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import {
  UnauthorizedError,
  getMe,
  login as apiLogin,
  logout as apiLogout,
  register as apiRegister,
  type AuthUser,
} from '../api/auth'
import { migrateLegacyStorage, setStorageUser } from '../lib/userStorage'

export type AuthStatus = 'loading' | 'anon' | 'authed' | 'offline'

interface AuthContextValue {
  status: AuthStatus
  user: AuthUser | null
  offlineMessage: string
  retry: () => Promise<void>
  signIn: (loginId: string, password: string) => Promise<void>
  signUp: (loginId: string, password: string) => Promise<void>
  signOut: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue | null>(null)

function applyUser(user: AuthUser) {
  setStorageUser(user.user_id)
  migrateLegacyStorage(user.user_id)
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>('loading')
  const [user, setUser] = useState<AuthUser | null>(null)
  const [offlineMessage, setOfflineMessage] = useState('')

  const refresh = useCallback(async () => {
    setStatus('loading')
    setOfflineMessage('')
    try {
      const current = await getMe()
      applyUser(current)
      setUser(current)
      setStatus('authed')
    } catch (error) {
      setUser(null)
      if (error instanceof UnauthorizedError) {
        setStatus('anon')
        return
      }
      setOfflineMessage(error instanceof Error ? error.message : '服务暂不可用')
      setStatus('offline')
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const signIn = useCallback(async (loginId: string, password: string) => {
    const current = await apiLogin(loginId, password)
    applyUser(current)
    setUser(current)
    setStatus('authed')
  }, [])

  const signUp = useCallback(async (loginId: string, password: string) => {
    const current = await apiRegister(loginId, password)
    applyUser(current)
    setUser(current)
    setStatus('authed')
  }, [])

  const signOut = useCallback(async () => {
    try {
      await apiLogout()
    } finally {
      setUser(null)
      setStatus('anon')
    }
  }, [])

  const value = useMemo<AuthContextValue>(
    () => ({
      status,
      user,
      offlineMessage,
      retry: refresh,
      signIn,
      signUp,
      signOut,
    }),
    [status, user, offlineMessage, refresh, signIn, signUp, signOut],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthContextValue {
  const value = useContext(AuthContext)
  if (!value) throw new Error('useAuth must be used inside AuthProvider')
  return value
}
