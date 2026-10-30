import {
  Fragment,
  useCallback,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type FocusEvent,
  type ReactNode,
  type Ref,
} from 'react'
import * as DropdownPrimitive from '@radix-ui/react-dropdown-menu'
import { ChevronDown } from 'lucide-react'

export interface DropdownItem {
  label: string
  icon?: ReactNode
  trailing?: ReactNode
  onSelect?: () => void
  disabled?: boolean
  destructive?: boolean
  separatorBefore?: boolean
  /** 传入即渲染为菜单内的复选项，勾选后菜单保持打开。 */
  checked?: boolean
  onCheckedChange?: (checked: boolean) => void
}

export interface DropdownMenuProps {
  label: string
  items: DropdownItem[]
  icon?: ReactNode
  triggerClassName?: string
  triggerRef?: Ref<HTMLButtonElement>
  /** 菜单关闭时是否把焦点还给触发器；打开对话框等场景可 preventDefault。 */
  onCloseAutoFocus?: (event: Event) => void
}

type Highlight = { top: number; height: number; danger: boolean; glide: boolean }

/**
 * uiarc.dev 的 dropdown menu 行为：Radix 菜单 + 跟随指针滑动的高亮块。
 * 本仓库没有 motion 依赖，滑动手感由 CSS 过渡实现。
 */
export function DropdownMenu({ label, items, icon, triggerClassName, triggerRef, onCloseAutoFocus }: DropdownMenuProps) {
  const [highlight, setHighlight] = useState<Highlight | null>(null)
  const pointer = useRef(false)
  const clearTimer = useRef(0)

  useEffect(() => () => window.clearTimeout(clearTimer.current), [])

  const measure = useCallback((element: HTMLElement) => {
    setHighlight(current => ({
      top: element.offsetTop,
      height: element.offsetHeight,
      danger: element.dataset.tone === 'danger',
      glide: pointer.current && current !== null,
    }))
  }, [])

  const handleItemFocus = (event: FocusEvent<HTMLDivElement>) => {
    window.clearTimeout(clearTimer.current)
    measure(event.currentTarget)
  }

  const handleContentFocus = (event: FocusEvent<HTMLDivElement>) => {
    const item = event.target instanceof HTMLElement
      ? event.target.closest<HTMLElement>('[role^="menuitem"]')
      : null
    window.clearTimeout(clearTimer.current)
    if (item) {
      measure(item)
      return
    }
    clearTimer.current = window.setTimeout(() => setHighlight(null), pointer.current ? 70 : 0)
  }

  return (
    <DropdownPrimitive.Root
      modal={false}
      onOpenChange={(open) => {
        if (open) {
          window.clearTimeout(clearTimer.current)
          setHighlight(null)
        }
      }}
    >
      <DropdownPrimitive.Trigger
        className={`inline-flex h-9 items-center gap-2 rounded-md border border-white/10 bg-black/50 px-3 text-xs font-medium text-white/80 backdrop-blur-md transition-colors duration-150 ease-[var(--ease-out)] hover:bg-black/60 data-[state=open]:bg-black/70 ${triggerClassName ?? ''}`}
        ref={triggerRef}
        type="button"
      >
        {icon && <span className="flex shrink-0 text-white/70" aria-hidden="true">{icon}</span>}
        <span className="truncate">{label}</span>
        <ChevronDown className="h-3.5 w-3.5 shrink-0 text-white/50" aria-hidden="true" />
      </DropdownPrimitive.Trigger>

      <DropdownPrimitive.Portal>
        <DropdownPrimitive.Content
          align="end"
          className="arc-menu"
          collisionPadding={12}
          loop
          onCloseAutoFocus={onCloseAutoFocus}
          onFocus={handleContentFocus}
          onPointerLeave={() => {
            pointer.current = false
            clearTimer.current = window.setTimeout(() => setHighlight(null), 70)
          }}
          onPointerMoveCapture={() => { pointer.current = true }}
          onKeyDownCapture={() => { pointer.current = false }}
          sideOffset={6}
        >
          <span
            aria-hidden="true"
            className="arc-menu-highlight"
            data-glide={highlight?.glide ? 'true' : 'false'}
            data-tone={highlight?.danger ? 'danger' : undefined}
            style={{
              height: highlight?.height ?? 0,
              opacity: highlight ? 1 : 0,
              transform: `translateY(${highlight?.top ?? 0}px)`,
            }}
          />
          {items.map((item, index) => {
            const itemClassName = `arc-menu-item${item.destructive ? ' arc-menu-item-danger' : ''}`
            const content = (
              <>
                {item.icon && <span className="arc-menu-icon" aria-hidden="true">{item.icon}</span>}
                <span className="min-w-0 flex-1 truncate">{item.label}</span>
                {item.trailing}
              </>
            )
            const style = { '--i': index } as CSSProperties
            const shared = {
              className: itemClassName,
              'data-tone': item.destructive ? 'danger' : undefined,
              disabled: item.disabled,
              style,
            }

            return (
              <Fragment key={item.label}>
                {item.separatorBefore && <DropdownPrimitive.Separator className="arc-menu-separator" />}
                {item.checked === undefined ? (
                  <DropdownPrimitive.Item
                    {...shared}
                    onFocus={handleItemFocus}
                    onPointerEnter={(event) => {
                      pointer.current = true
                      measure(event.currentTarget)
                    }}
                    onSelect={item.onSelect}
                  >
                    {content}
                  </DropdownPrimitive.Item>
                ) : (
                  <DropdownPrimitive.CheckboxItem
                    {...shared}
                    checked={item.checked}
                    onCheckedChange={(next) => item.onCheckedChange?.(next === true)}
                    onFocus={handleItemFocus}
                    onPointerEnter={(event) => {
                      pointer.current = true
                      measure(event.currentTarget)
                    }}
                    onSelect={(event) => event.preventDefault()}
                  >
                    {content}
                  </DropdownPrimitive.CheckboxItem>
                )}
              </Fragment>
            )
          })}
        </DropdownPrimitive.Content>
      </DropdownPrimitive.Portal>
    </DropdownPrimitive.Root>
  )
}

export default DropdownMenu
