import { Routes, Route, NavLink, Navigate, useLocation } from 'react-router-dom'
import {
  Mountain,
  Home,
  Search,
  Bot,
  Settings,
} from 'lucide-react'
import HomePage from './pages/Home'
import HikingRAG from './pages/HikingRAG'
import SuperAgent from './pages/SuperAgent'
import LlmConfig from './pages/LlmConfig'

function App() {
  const location = useLocation()

  const getHeaderTitle = () => {
    if (location.pathname === '/love-master' || location.pathname === '/hiking-rag') return 'RAG 模块'
    if (location.pathname === '/super-agent') return 'Agent 模块'
    if (location.pathname === '/llm-config') return 'LLM 配置'
    return '智能徒步助手'
  }

  const isHome = location.pathname === '/'

  return (
    <div className="relative flex h-[100dvh] w-full bg-bg-body overflow-hidden">
      {/* Top Dock */}
      <nav className="fixed top-6 left-1/2 -translate-x-1/2 z-[100] flex items-center gap-2 p-2 rounded-full bg-black/40 backdrop-blur-md border border-white/10 shadow-2xl transition-colors duration-150 ease-[var(--ease-out)] hover:bg-black/50">
        <div className="flex items-center justify-center w-10 h-10 shrink-0 ml-2 mr-4">
          <Mountain className="w-6 h-6 text-white" strokeWidth={2} />
        </div>
        <div className="flex items-center gap-1 pr-2">
          <NavLink
            to="/"
            className={({ isActive }) =>
              `flex items-center justify-center w-10 h-10 rounded-full transition-[background-color,color,box-shadow] duration-150 ease-[var(--ease-out)] ${
                isActive ? 'bg-white/20 text-white shadow-sm' : 'text-white/70 hover:bg-white/10 hover:text-white'
              }`
            }
            title="首页"
            end
          >
            <Home className="w-5 h-5" strokeWidth={2} />
          </NavLink>
          <NavLink
            to="/hiking-rag"
            className={({ isActive }) =>
              `flex items-center justify-center w-10 h-10 rounded-full transition-[background-color,color,box-shadow] duration-150 ease-[var(--ease-out)] ${
                isActive ? 'bg-white/20 text-white shadow-sm' : 'text-white/70 hover:bg-white/10 hover:text-white'
              }`
            }
            title="RAG 模块"
          >
            <Search className="w-5 h-5" strokeWidth={2} />
          </NavLink>
          <NavLink
            to="/super-agent"
            className={({ isActive }) =>
              `flex items-center justify-center w-10 h-10 rounded-full transition-[background-color,color,box-shadow] duration-150 ease-[var(--ease-out)] ${
                isActive ? 'bg-white/20 text-white shadow-sm' : 'text-white/70 hover:bg-white/10 hover:text-white'
              }`
            }
            title="Agent 模块"
          >
            <Bot className="w-5 h-5" strokeWidth={2} />
          </NavLink>
        </div>
      </nav>

      {/* Main Content */}
      <div className="flex-1 flex flex-col w-full h-full relative overflow-hidden">


        <main className="flex-1 overflow-y-auto flex flex-col relative w-full h-full">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/love-master" element={<Navigate to="/hiking-rag" replace />} />
            <Route path="/hiking-rag" element={<HikingRAG />} />
            <Route path="/super-agent" element={<SuperAgent />} />
            <Route path="/llm-config" element={<LlmConfig />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}

export default App
