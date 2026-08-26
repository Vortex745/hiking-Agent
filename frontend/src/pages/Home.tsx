import { Link } from 'react-router-dom'
import { Search, Bot, ArrowRight } from 'lucide-react'
import { useRef, useState } from 'react'
import gsap from 'gsap'
import { useGSAP } from '@gsap/react'
import { TextPlugin } from 'gsap/TextPlugin'

gsap.registerPlugin(useGSAP, TextPlugin)

function Home() {
  const containerRef = useRef<HTMLDivElement>(null)
  const titleRef = useRef<HTMLHeadingElement>(null)
  const [videoReady, setVideoReady] = useState(false)

  useGSAP(() => {
    // 文字数量缩减然后再增加的无限循环打字机效果
    gsap.timeline({ repeat: -1, repeatDelay: 3 })
      .to(titleRef.current, {
        text: { value: "AI" },
        duration: 0.8,
        ease: "none",
        delay: 1
      })
      .to(titleRef.current, {
        text: { value: "AI智能徒步助手" },
        duration: 1.2,
        ease: "none",
        delay: 0.5
      })
  }, { scope: containerRef })

  return (
    <div ref={containerRef} className="relative flex flex-col items-center justify-center min-h-screen bg-gray-950 overflow-hidden pt-20">
      {/* Background Video */}
      <video
        aria-hidden="true"
        autoPlay
        loop
        muted
        playsInline
        preload="auto"
        onPlaying={() => setVideoReady(true)}
        className={`absolute inset-0 w-full h-full object-cover z-[1] pointer-events-none transition-opacity duration-700 ${
          videoReady ? 'opacity-100' : 'opacity-0'
        }`}
        src="/ascii-art.mp4"
      />
      {/* Dark overlay for text contrast */}
      <div className="absolute inset-0 bg-black/50 z-[1] pointer-events-none" />

      {/* Content */}
      <div className="relative z-[2] isolate animate-[fadeIn_400ms_var(--ease-out)_both] px-6 max-w-[1000px] w-full flex flex-col items-center justify-center flex-1">
        {/* Hero */}
        <div className="text-center mb-16 h-[80px] flex items-center justify-center">
          <h1 ref={titleRef} className="text-5xl md:text-7xl font-['Noto_Sans_SC',sans-serif] font-black tracking-tight text-white drop-shadow-2xl">AI智能徒步助手</h1>
        </div>

        {/* Module Cards */}
        <div className="grid grid-cols-2 gap-8 w-full max-w-[800px] max-md:grid-cols-1">
          <Link
            to="/hiking-rag"
            className="group flex flex-col bg-white/10 backdrop-blur-xl rounded-[24px] p-8 border border-white/20 transition-[transform,background-color,border-color,box-shadow] duration-200 ease-[var(--ease-out)] hover:-translate-y-1 hover:bg-white/20 hover:border-white/40 hover:shadow-[0_8px_32px_0_rgba(0,0,0,0.3)] cursor-pointer"
            style={{ animation: 'fadeUp 260ms var(--ease-out) 50ms both' }}
          >
            <div className="flex items-center gap-4 mb-5">
              <div className="w-14 h-14 rounded-full bg-white/20 text-white flex items-center justify-center text-2xl transition-[transform,background-color] duration-200 ease-[var(--ease-out)] group-hover:scale-105 group-hover:bg-white/30 shadow-inner">
                <Search className="w-7 h-7" strokeWidth={1.5} />
              </div>
              <div>
                <div className="text-xl font-bold text-white">RAG 模块</div>
                <div className="inline-block px-3 py-1 bg-white/20 text-white text-[11px] font-medium tracking-wide rounded-full mt-2 backdrop-blur-sm border border-white/10">对话式界面</div>
              </div>
            </div>
            <p className="text-[15px] text-white/90 mb-8 leading-relaxed flex-1 font-medium">
              知识聊天助手，专注于徒步知识问答与检索。
            </p>
            <div className="mt-auto inline-flex items-center gap-2 text-white text-[15px] font-bold">
              进入对话
              <ArrowRight className="w-5 h-5 transition-transform duration-[160ms] ease-[var(--ease-out)] group-hover:translate-x-1" />
            </div>
          </Link>

          <Link
            to="/super-agent"
            className="group flex flex-col bg-white/10 backdrop-blur-xl rounded-[24px] p-8 border border-white/20 transition-[transform,background-color,border-color,box-shadow] duration-200 ease-[var(--ease-out)] hover:-translate-y-1 hover:bg-white/20 hover:border-white/40 hover:shadow-[0_8px_32px_0_rgba(0,0,0,0.3)] cursor-pointer"
            style={{ animation: 'fadeUp 260ms var(--ease-out) 110ms both' }}
          >
            <div className="flex items-center gap-4 mb-5">
              <div className="w-14 h-14 rounded-full bg-white/20 text-white flex items-center justify-center text-2xl transition-[transform,background-color] duration-200 ease-[var(--ease-out)] group-hover:scale-105 group-hover:bg-white/30 shadow-inner">
                <Bot className="w-7 h-7" strokeWidth={1.5} />
              </div>
              <div>
                <div className="text-xl font-bold text-white">Agent 模块</div>
                <div className="inline-block px-3 py-1 bg-white/20 text-white text-[11px] font-medium tracking-wide rounded-full mt-2 backdrop-blur-sm border border-white/10">对话式界面</div>
              </div>
            </div>
            <p className="text-[15px] text-white/90 mb-8 leading-relaxed flex-1 font-medium">
              行动/规划聊天助手，专注于行程规划与执行。
            </p>
            <div className="mt-auto inline-flex items-center gap-2 text-white text-[15px] font-bold">
              进入对话
              <ArrowRight className="w-5 h-5 transition-transform duration-[160ms] ease-[var(--ease-out)] group-hover:translate-x-1" />
            </div>
          </Link>
        </div>
      </div>
    </div>
  )
}

export default Home
