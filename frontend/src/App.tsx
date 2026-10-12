import { Routes, Route } from 'react-router-dom'
import SuperAgent from './pages/SuperAgent'
import LlmConfig from './pages/LlmConfig'

function App() {
  return (
    <div className="relative flex h-[100dvh] w-full bg-bg-body overflow-hidden">
      {/* Main Content */}
      <div className="flex-1 flex flex-col w-full h-full relative overflow-hidden">


        <main className="flex-1 overflow-y-auto flex flex-col relative w-full h-full">
          <Routes>
            <Route path="/" element={<SuperAgent />} />
            <Route path="/llm-config" element={<LlmConfig />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}

export default App
