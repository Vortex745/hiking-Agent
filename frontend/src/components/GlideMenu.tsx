import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
} from 'react';

/* GlideMenu — 一列可选项，悬停/聚焦时一块高亮背景在行间「滑行」。
 * 行用 data-menu-row 标记；高亮纯粹是装饰，pointer-events 关闭。
 * 与项目下拉/菜单共用同一条 ease 曲线 cubic-bezier(0.22,1,0.36,1)。 */

type GlideMenuProps = {
  children: ReactNode;
  className?: string;
  highlightClassName?: string;
};

type Rect = { top: number; height: number };

export default function GlideMenu({ children, className = '', highlightClassName = '' }: GlideMenuProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const activeRef = useRef<HTMLElement | null>(null);
  const [rect, setRect] = useState<Rect | null>(null);

  const measure = useCallback((el: HTMLElement | null) => {
    activeRef.current = el;
    if (!el) {
      setRect(null);
      return;
    }
    setRect({ top: el.offsetTop, height: el.offsetHeight });
  }, []);

  const handleMove = useCallback(
    (target: EventTarget | null) => {
      const row = (target as HTMLElement | null)?.closest?.<HTMLElement>('[data-menu-row]');
      // 只在切换行时更新，避免同一行内的移动反复 setState
      if (row === activeRef.current) return;
      measure(row ?? null);
    },
    [measure],
  );

  // 容器或行尺寸变化时（内容换行、切题重排）重新量当前高亮行
  useLayoutEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const ro = new ResizeObserver(() => {
      if (activeRef.current) measure(activeRef.current);
    });
    ro.observe(container);
    return () => ro.disconnect();
  }, [measure]);

  useEffect(() => () => {
    activeRef.current = null;
  }, []);

  return (
    <div
      ref={containerRef}
      className={`relative ${className}`.trim()}
      onPointerOver={(e) => handleMove(e.target)}
      onPointerLeave={() => measure(null)}
      onFocus={(e) => handleMove(e.target)}
      onBlur={(e) => {
        if (!containerRef.current?.contains(e.relatedTarget as Node | null)) measure(null);
      }}
    >
      <span
        aria-hidden
        className={`pointer-events-none absolute left-0 top-0 transition-all duration-200 ease-[cubic-bezier(0.22,1,0.36,1)] ${
          rect ? 'opacity-100' : 'opacity-0'
        } ${highlightClassName}`.trim()}
        style={{
          height: rect?.height ?? 0,
          transform: `translate3d(0, ${rect?.top ?? 0}px, 0)`,
          willChange: 'transform, height',
        }}
      />
      {children}
    </div>
  );
}
