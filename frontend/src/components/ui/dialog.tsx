import { useRef, type ReactNode } from 'react'
import { Dialog, DialogContent } from '../arc/dialog/dialog'

export type ArcDialogSize = 'sm' | 'md' | 'lg' | 'xl'

export interface ArcDialogProps {
  title: string
  description?: string
  actions?: ReactNode
  footer?: ReactNode
  children: ReactNode
  onClose: () => void
  size?: ArcDialogSize
  initialFocus?: 'close' | 'first'
}

export function ArcDialog({ title, description, actions, footer, children, onClose, size = 'md', initialFocus = 'close' }: ArcDialogProps) {
  const bodyRef = useRef<HTMLDivElement>(null)
  const previousFocus = useRef<HTMLElement | null>(null)

  return (
    <Dialog open onOpenChange={open => { if (!open) onClose() }}>
      <DialogContent
        actions={actions}
        data-size={size}
        description={description}
        footer={footer}
        onOpenAutoFocus={event => {
          previousFocus.current = document.activeElement instanceof HTMLElement ? document.activeElement : null
          const first = initialFocus === 'first'
            ? bodyRef.current?.querySelector<HTMLElement>('button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])')
            : event.target instanceof HTMLElement ? event.target.querySelector<HTMLElement>('button[aria-label="关闭"]') : null
          if (first) {
            event.preventDefault()
            first.focus()
          }
        }}
        onCloseAutoFocus={event => {
          if (previousFocus.current?.isConnected) {
            event.preventDefault()
            previousFocus.current.focus()
          }
        }}
        title={title}
      >
        <div ref={bodyRef}>{children}</div>
      </DialogContent>
    </Dialog>
  )
}

export default ArcDialog
