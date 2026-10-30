import type { ReactNode } from 'react'

export interface ArcCardProps {
  title: string
  description?: string
  media?: ReactNode
  meta?: ReactNode
  status?: string
  action?: ReactNode
  onOpen: () => void
  children?: ReactNode
  className?: string
}

/**
 * Arc 风格卡片：整张卡片由标题按钮铺满命中区域，点击后由调用方打开对话框。
 */
export function ArcCard({
  title,
  description,
  media,
  meta,
  status,
  action,
  onOpen,
  children,
  className,
}: ArcCardProps) {
  return (
    <article className={`arc-card ${className ?? ''}`}>
      {media && <div className="arc-card-media">{media}</div>}
      <div className="arc-card-content">
        <h3 className="arc-card-title">
          <button className="arc-card-trigger" onClick={onOpen} type="button">{title}</button>
        </h3>
        {description && <p className="arc-card-description">{description}</p>}
        {children}
        {(meta || status || action) && (
          <div className="arc-card-footer">
            <div className="arc-card-byline">
              {meta && <span className="arc-card-meta">{meta}</span>}
              {status && <span className="arc-card-status">{status}</span>}
            </div>
            {action && <div className="arc-card-action">{action}</div>}
          </div>
        )}
      </div>
    </article>
  )
}

export default ArcCard
