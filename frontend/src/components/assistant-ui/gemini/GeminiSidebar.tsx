import { Menu, MessageSquare, PanelLeft, Plus, Trash2, X } from 'lucide-react'

export interface ChatSessionMeta {
  id: string
  title: string
  date: string
}

interface GeminiSidebarProps {
  /** Desktop: rail (48px) vs expanded list (260px) */
  collapsed: boolean
  onCollapsedChange: (collapsed: boolean) => void
  /** Mobile: sheet visibility */
  mobileOpen: boolean
  onMobileOpenChange: (open: boolean) => void
  sessions: ChatSessionMeta[]
  activeSessionId: string | null
  onNewSession: () => void
  onSelectSession: (sessionId: string) => void
  onDeleteSession: (sessionId: string) => void
}

const ghostBtnClass =
  'flex size-8 shrink-0 items-center justify-center rounded-lg text-[#444746] transition-colors hover:bg-black/5 dark:text-[#c4c7c5] dark:hover:bg-white/10'

/**
 * Gemini-style docked session sidebar, mirroring assistant-ui's
 * CloneThreadShell: desktop renders an inline rail that collapses to a
 * 48px icon strip; mobile renders a trigger button plus a left sheet.
 * The parent owns all session state — this component only reads
 * id/title/date and reports intents back.
 */
export const GeminiSidebar = ({
  collapsed,
  onCollapsedChange,
  mobileOpen,
  onMobileOpenChange,
  sessions,
  activeSessionId,
  onNewSession,
  onSelectSession,
  onDeleteSession,
}: GeminiSidebarProps) => {
  const closeMobile = () => onMobileOpenChange(false)

  const newThreadButton = (collapsedStyle: boolean, onNavigate?: () => void) => (
    <button
      type="button"
      onClick={() => {
        onNewSession()
        onNavigate?.()
      }}
      aria-label="新建对话"
      title="新建对话"
      className={`flex h-8 shrink-0 items-center overflow-hidden rounded-lg text-sm text-[#444746] transition-[width,background-color] duration-200 hover:bg-black/5 dark:text-[#c4c7c5] dark:hover:bg-white/10 ${
        collapsedStyle ? 'w-8 justify-center' : 'w-full gap-2 px-2.5'
      }`}
    >
      <Plus className="size-4 shrink-0" />
      <span
        className={`overflow-hidden whitespace-nowrap transition-all duration-200 ${
          collapsedStyle ? 'max-w-0 opacity-0' : 'max-w-24 opacity-100'
        }`}
      >
        新建对话
      </span>
    </button>
  )

  const sessionItems = (onNavigate?: () => void) => (
    <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-2 scrollbar-thin">
      {sessions.length === 0 && (
        <div className="px-2 py-6 text-center text-xs text-[#5e6063] dark:text-[#9aa0a6]">暂无历史对话</div>
      )}
      <div className="flex flex-col gap-0.5">
        {sessions.map((session) => {
          const active = session.id === activeSessionId
          return (
            <div
              key={session.id}
              onClick={() => {
                onSelectSession(session.id)
                onNavigate?.()
              }}
              className={`group flex cursor-pointer items-center gap-2.5 rounded-xl px-2.5 py-2 transition-colors ${
                active
                  ? 'bg-[#d3e3fd] text-[#001d35] dark:bg-[#004a77] dark:text-[#c2e7ff]'
                  : 'text-[#1f1f1f] hover:bg-black/5 dark:text-[#e3e3e3] dark:hover:bg-white/10'
              }`}
            >
              <MessageSquare className="size-4 shrink-0 opacity-50" />
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm leading-5">{session.title}</div>
                <div className="truncate text-[11px] leading-4 opacity-60">{session.date}</div>
              </div>
              <button
                type="button"
                aria-label="删除对话"
                title="删除对话"
                onClick={(e) => {
                  e.stopPropagation()
                  onDeleteSession(session.id)
                }}
                className="shrink-0 rounded-full p-1.5 text-[#444746] opacity-0 transition-[opacity,color,background-color] hover:bg-black/10 hover:text-[#b3261e] focus-visible:opacity-100 group-hover:opacity-100 dark:text-[#c4c7c5] dark:hover:bg-white/15 dark:hover:text-[#f2b8b5]"
              >
                <Trash2 className="size-3.5" />
              </button>
            </div>
          )
        })}
      </div>
    </div>
  )

  return (
    <>
      {/* Desktop: docked rail, collapses to a 48px icon strip */}
      <aside
        aria-label="对话历史"
        className={`hidden h-full shrink-0 flex-col overflow-hidden border-r border-black/5 bg-[#f0f4f9] transition-[width] duration-200 ease-[var(--ease-out)] md:flex dark:border-white/10 dark:bg-[#1e1f20] ${
          collapsed ? 'w-12' : 'w-[260px]'
        }`}
      >
        <div className="flex h-12 shrink-0 items-center overflow-hidden px-2">
          <button
            type="button"
            onClick={() => onCollapsedChange(!collapsed)}
            aria-label={collapsed ? '展开侧边栏' : '收起侧边栏'}
            aria-expanded={!collapsed}
            title={collapsed ? '展开侧边栏' : '收起侧边栏'}
            className={ghostBtnClass}
          >
            <PanelLeft className="size-4" />
          </button>
          {!collapsed && (
            <span className="ml-2 truncate text-sm font-medium text-[#1f1f1f] dark:text-[#e3e3e3]">对话历史</span>
          )}
        </div>

        {/* New thread stays reachable as an icon when the rail is collapsed */}
        <div className="flex min-h-0 flex-1 flex-col px-2 pb-2 pt-1">
          {newThreadButton(collapsed)}

          <div
            aria-hidden={collapsed}
            className={`mt-1 flex min-h-0 flex-1 flex-col transition-[opacity,transform] duration-150 ${
              collapsed ? 'pointer-events-none translate-x-1 opacity-0' : 'translate-x-0 opacity-100'
            }`}
          >
            {sessionItems()}
          </div>
        </div>
      </aside>

      {/* Mobile trigger — top-left ghost button, like the example's SheetTrigger */}
      <div className="absolute top-2 left-2 z-20 md:hidden">
        <button
          type="button"
          onClick={() => onMobileOpenChange(true)}
          aria-label="打开对话历史"
          title="对话历史"
          className="flex size-8 items-center justify-center rounded-lg bg-[#fdfcfc]/70 text-[#444746] backdrop-blur-sm transition-colors hover:bg-[#fdfcfc] dark:bg-[#1e1f20]/70 dark:text-[#c4c7c5] dark:hover:bg-[#1e1f20]"
        >
          <Menu className="size-4" />
        </button>
      </div>

      {/* Mobile backdrop — above the app's fixed top dock (z-100) so the
          open sheet behaves as a full modal */}
      <div
        aria-hidden="true"
        onClick={closeMobile}
        className={`fixed inset-0 z-[110] bg-black/40 backdrop-blur-sm transition-opacity duration-200 md:hidden ${
          mobileOpen ? 'opacity-100' : 'pointer-events-none opacity-0'
        }`}
      />

      {/* Mobile sheet */}
      <aside
        aria-label="对话历史"
        aria-hidden={!mobileOpen}
        className={`fixed inset-y-0 left-0 z-[111] flex w-[280px] max-w-[85vw] flex-col bg-[#f0f4f9] shadow-xl transition-transform duration-200 ease-[var(--ease-out)] md:hidden dark:bg-[#1e1f20] ${
          mobileOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex h-12 shrink-0 items-center justify-between px-3">
          <span className="text-sm font-medium text-[#1f1f1f] dark:text-[#e3e3e3]">对话历史</span>
          <button type="button" onClick={closeMobile} aria-label="关闭对话历史" className={ghostBtnClass}>
            <X className="size-4" />
          </button>
        </div>
        <div className="flex min-h-0 flex-1 flex-col px-2 pb-3 pt-1">
          {newThreadButton(false, closeMobile)}
          <div className="mt-1 flex min-h-0 flex-1 flex-col">{sessionItems(closeMobile)}</div>
        </div>
      </aside>
    </>
  )
}
