import { AnimatePresence, animate, motion, useMotionValue, useReducedMotion } from 'motion/react'
import { LoaderCircle, Map as MapIcon, MapPin, X } from 'lucide-react'
import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import { useAuth } from '../../auth/AuthContext'
import { AMAP_CONFIG_CHANGED, getAMapBrowserConfig, type AMapBrowserConfig } from '../../api/mcp'

export interface ViewOnMapFocus {
  name: string
}

export interface ViewOnMapProps {
  /** 当前会话由 AI 识别的唯一景点；为空时默认展示当前位置。 */
  focus?: ViewOnMapFocus | null
  /** 会话标识变化时重新解析目标。 */
  sessionKey?: string
  className?: string
}

type ResolveState = 'idle' | 'loading' | 'ready' | 'error' | 'unconfigured'

type ResolvedTarget = {
  kind: 'spot' | 'current' | 'fallback'
  name: string
  lng: number
  lat: number
  zoom: number
  accuracy?: number
}

interface AMapMap {
  on(event: string, handler: () => void): void
  add(overlay: unknown): void
  resize(): void
  destroy(): void
}

interface AMapPoi {
  name?: string
  typecode?: string
  location?: unknown
}

interface AMapPlaceSearch {
  search(
    keyword: string,
    callback: (status: string, result: AMapPlaceSearchResult | undefined) => void,
  ): void
}

interface AMapPlaceSearchResult {
  poiList?: { pois?: AMapPoi[] }
}

interface AMapGeolocation {
  getCurrentPosition(
    callback: (status: string, result: AMapGeolocationResult | undefined) => void,
  ): void
}

interface AMapGeolocationResult {
  position?: { lng: number; lat: number }
  accuracy?: number
}

interface AMapNamespace {
  Map: new (container: HTMLElement, options: Record<string, unknown>) => AMapMap
  Marker: new (options: Record<string, unknown>) => unknown
  Circle: new (options: Record<string, unknown>) => unknown
  PlaceSearch: new (options?: Record<string, unknown>) => AMapPlaceSearch
  Geolocation: new (options?: Record<string, unknown>) => AMapGeolocation
}

const FALLBACK_BG =
  'radial-gradient(circle at 30% 20%, color-mix(in srgb, var(--primary-light) 14%, transparent), transparent 62%)'
const FALLBACK_CENTER = { lng: 104.195397, lat: 35.86166 }

function readLngLat(value: unknown): { lng: number; lat: number } | null {
  if (Array.isArray(value) && value.length >= 2 && typeof value[0] === 'number' && typeof value[1] === 'number') {
    return { lng: value[0], lat: value[1] }
  }
  if (value && typeof value === 'object') {
    const lng = (value as { lng?: unknown }).lng
    const lat = (value as { lat?: unknown }).lat
    if (typeof lng === 'number' && typeof lat === 'number') return { lng, lat }
  }
  return null
}

function searchScenicSpot(AMap: AMapNamespace, keyword: string): Promise<{ name: string; lng: number; lat: number } | null> {
  return new Promise(resolve => {
    // type=110000 已限定风景名胜类；搜索结果通常不返回 typecode，不能再按字段过滤。
    const search = new AMap.PlaceSearch({ type: '110000', pageSize: 10 })
    search.search(keyword, (status, result) => {
      if (status !== 'complete') {
        resolve(null)
        return
      }
      const pois = result?.poiList?.pois ?? []
      const candidates = pois
        .map(poi => ({ poi, point: readLngLat(poi.location) }))
        .filter((item): item is { poi: AMapPoi; point: { lng: number; lat: number } } => item.point !== null)
      if (candidates.length === 0) {
        resolve(null)
        return
      }
      const keywordCore = keyword.replace(/(国家级)?风景名胜区|景区|公园/g, '').trim()
      const matched = keywordCore
        ? candidates.find(item => typeof item.poi.name === 'string' && item.poi.name.includes(keywordCore))
        : undefined
      const first = matched ?? candidates[0]
      resolve({
        name: typeof first.poi.name === 'string' && first.poi.name ? first.poi.name : keyword,
        lng: first.point.lng,
        lat: first.point.lat,
      })
    })
  })
}

function locateCurrent(AMap: AMapNamespace): Promise<{ lng: number; lat: number; accuracy?: number } | null> {
  return new Promise(resolve => {
    const geolocation = new AMap.Geolocation({ enableHighAccuracy: true, timeout: 10000 })
    geolocation.getCurrentPosition((status, result) => {
      const position = status === 'complete' ? result?.position : undefined
      resolve(position ? { lng: position.lng, lat: position.lat, accuracy: result?.accuracy } : null)
    })
  })
}

function MapCanvas({ config, focusName, isDark, onTarget, onState, onClose }: {
  config: AMapBrowserConfig
  focusName: string | null
  isDark: boolean
  onTarget: (target: ResolvedTarget) => void
  onState: (state: ResolveState) => void
  onClose: () => void
}) {
  const frameRef = useRef<HTMLIFrameElement>(null)
  const [frameLoaded, setFrameLoaded] = useState(false)

  useLayoutEffect(() => {
    const frameWindow = frameRef.current?.contentWindow as (Window & {
      AMap?: AMapNamespace
      _AMapSecurityConfig?: { securityJsCode: string }
      onAMapLoaded?: () => void
    }) | null
    const doc = frameWindow?.document
    if (!frameLoaded || !frameWindow || !doc) return

    let disposed = false
    let map: AMapMap | null = null
    let observer: ResizeObserver | null = null
    let readyTimer = 0
    onState('loading')
    const onKeyDown = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose() }
    doc.addEventListener('keydown', onKeyDown)
    // SDK globals live only in this frame. Removing the frame releases its Key and location.
    frameWindow._AMapSecurityConfig = { securityJsCode: config.security_code }
    const container = doc.createElement('div')
    container.style.cssText = 'position:absolute;inset:0'
    doc.body.appendChild(container)
    const script = doc.createElement('script')
    const timeout = window.setTimeout(() => {
      if (!disposed) onState('error')
    }, 20000)
    const fail = () => {
      window.clearTimeout(timeout)
      if (!disposed) onState('error')
    }
    frameWindow.onAMapLoaded = () => {
      void (async () => {
        try {
          const AMap = frameWindow.AMap!
          let target: ResolvedTarget
          if (focusName) {
            const spot = await searchScenicSpot(AMap, focusName)
            if (disposed) return
            if (!spot) { fail(); return }
            target = { kind: 'spot', ...spot, zoom: 14 }
          } else {
            const current = await locateCurrent(AMap)
            if (disposed) return
            target = current
              ? { kind: 'current', name: '当前位置', ...current, zoom: 15 }
              : { kind: 'fallback', name: '定位未授权', ...FALLBACK_CENTER, zoom: 4 }
          }
          onTarget(target)
          map = new AMap.Map(container, {
            viewMode: '2D', zoom: target.zoom, center: [target.lng, target.lat],
            mapStyle: isDark ? 'amap://styles/dark' : 'amap://styles/normal',
          })
          observer = new ResizeObserver(() => map?.resize())
          observer.observe(container)
          if (target.kind !== 'fallback') {
            if (target.kind === 'current' && target.accuracy && target.accuracy > 0) {
              map.add(new AMap.Circle({
                center: [target.lng, target.lat], radius: target.accuracy,
                strokeColor: '#1a73e8', strokeOpacity: 0.35, strokeWeight: 1,
                fillColor: '#1a73e8', fillOpacity: 0.12,
              }))
            }
            map.add(new AMap.Marker({ position: [target.lng, target.lat], title: target.name }))
          }
          const ready = () => {
            window.clearTimeout(timeout)
            if (!disposed) onState('ready')
          }
          map.on('tilesloaded', ready)
          map.on('complete', () => { readyTimer = window.setTimeout(ready, 900) })
        } catch { fail() }
      })()
    }
    script.src = `https://webapi.amap.com/maps?${new URLSearchParams({
      v: '2.0', key: config.key, plugin: 'AMap.PlaceSearch,AMap.Geolocation', callback: 'onAMapLoaded',
    })}`
    script.onerror = fail
    doc.head.appendChild(script)
    return () => {
      disposed = true
      window.clearTimeout(timeout)
      window.clearTimeout(readyTimer)
      delete frameWindow.onAMapLoaded
      doc.removeEventListener('keydown', onKeyDown)
      observer?.disconnect()
      map?.destroy()
      script.remove()
      container.remove()
    }
  }, [config, focusName, frameLoaded, isDark, onClose, onState, onTarget])

  return <iframe ref={frameRef} title="高德地图" className="absolute inset-0 h-full w-full border-0"
    srcDoc="<!doctype html><html><head><meta charset='utf-8'></head><body></body></html>"
    allow="geolocation" onLoad={() => setFrameLoaded(true)} />
}

const BALL_SIZE = 56
const BALL_MARGIN = 12
const MAGNET_ZONE = 56
const MAGNET_PULL = 8
const SNAP_THRESHOLD = 40
const POSITION_STORAGE_KEY = 'view-on-map:ball-position'

type BallPosition = { x: number; y: number }

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max)
}

function clampPosition(position: BallPosition, width: number, height: number): BallPosition {
  return {
    x: clamp(position.x, BALL_MARGIN, Math.max(BALL_MARGIN, width - BALL_SIZE - BALL_MARGIN)),
    y: clamp(position.y, BALL_MARGIN, Math.max(BALL_MARGIN, height - BALL_SIZE - BALL_MARGIN)),
  }
}

function defaultPosition(width: number, height: number): BallPosition {
  return clampPosition(
    { x: width - BALL_SIZE - (width < 640 ? 16 : 24), y: width < 640 ? 96 : 240 },
    width,
    height,
  )
}

function initialPosition(): BallPosition {
  const width = window.innerWidth
  const height = window.innerHeight
  const fallback = defaultPosition(width, height)
  try {
    const raw = window.localStorage.getItem(POSITION_STORAGE_KEY)
    if (!raw) return fallback
    const parsed = JSON.parse(raw) as Partial<BallPosition> | null
    if (typeof parsed?.x === 'number' && typeof parsed?.y === 'number') {
      // 桌面端保存的位置在窄屏会压住标题，直接改用移动端默认位。
      if (width < 640 && (parsed.x + BALL_SIZE > width || parsed.y + BALL_SIZE > height)) {
        return fallback
      }
      return clampPosition({ x: parsed.x, y: parsed.y }, width, height)
    }
  } catch {
    // 本地存储不可用时退回默认位置
  }
  return fallback
}

type Edge = 'left' | 'right' | 'top' | 'bottom'

function nearestEdge(position: BallPosition, width: number, height: number) {
  const distances = [
    { edge: 'left' as const, distance: position.x },
    { edge: 'right' as const, distance: width - BALL_SIZE - position.x },
    { edge: 'top' as const, distance: position.y },
    { edge: 'bottom' as const, distance: height - BALL_SIZE - position.y },
  ]
  return distances.reduce((current, item) => (item.distance < current.distance ? item : current))
}

function magnetPull(position: BallPosition, width: number, height: number) {
  const nearest = nearestEdge(position, width, height)
  if (nearest.distance > MAGNET_ZONE) return { dx: 0, dy: 0 }
  const strength = (MAGNET_ZONE - nearest.distance) / MAGNET_ZONE
  const pull = Math.max(0, strength * MAGNET_PULL)
  switch (nearest.edge) {
    case 'left':
      return { dx: -pull, dy: 0 }
    case 'right':
      return { dx: pull, dy: 0 }
    case 'top':
      return { dx: 0, dy: -pull }
    case 'bottom':
      return { dx: 0, dy: pull }
    default:
      return { dx: 0, dy: 0 }
  }
}

function snapToEdge(position: BallPosition, width: number, height: number): BallPosition {
  const nearest = nearestEdge(position, width, height)
  if (nearest.distance > SNAP_THRESHOLD) return position
  switch (nearest.edge) {
    case 'left':
      return { ...position, x: BALL_MARGIN }
    case 'right':
      return { ...position, x: width - BALL_SIZE - BALL_MARGIN }
    case 'top':
      return { ...position, y: BALL_MARGIN }
    case 'bottom':
      return { ...position, y: height - BALL_SIZE - BALL_MARGIN }
    default:
      return position
  }
}

/**
 * 可拖拽的地图圆球：只显示图标，正常拖动 1:1 跟手，靠近视口边缘时磁吸并阻尼回弹。
 * 点击圆球在附近展开地图；有 AI 唯一景点时优先展示景点，否则首次打开请求当前位置。
 */
export function ViewOnMap({ focus, sessionKey, className }: ViewOnMapProps) {
  const { user } = useAuth()
  const [accountConfig, setConfig] = useState<{ ownerId: string; value: AMapBrowserConfig } | null>(null)
  const config = accountConfig?.ownerId === user?.user_id ? accountConfig?.value : null
  const [configRevision, setConfigRevision] = useState(0)
  const [isOpen, setIsOpen] = useState(false)
  const [isDark, setIsDark] = useState(() => window.matchMedia('(prefers-color-scheme: dark)').matches)
  const [viewport, setViewport] = useState(() => ({ width: window.innerWidth, height: window.innerHeight }))
  const [initial] = useState(initialPosition)
  const x = useMotionValue(initial.x)
  const y = useMotionValue(initial.y)
  const basePositionRef = useRef(initial)
  const [cardPosition, setCardPosition] = useState<BallPosition>(initial)
  const [isDragging, setIsDragging] = useState(false)
  const [target, setTarget] = useState<ResolvedTarget | null>(null)
  const [resolveState, setResolveState] = useState<ResolveState>('idle')
  const [pulse, setPulse] = useState(false)
  const focusName = focus?.name?.trim() || null
  const ballRef = useRef<HTMLButtonElement>(null)
  const closeRef = useRef<HTMLButtonElement>(null)
  const dragRef = useRef<{
    pointerId: number
    offsetX: number
    offsetY: number
    startX: number
    startY: number
    moved: boolean
  } | null>(null)
  const draggedRef = useRef(false)
  const snapControlsRef = useRef<Array<{ stop: () => void }>>([])
  const previousFocusRef = useRef<string | null>(focusName)
  const reduced = useReducedMotion()
  const id = useId()
  const panelId = `view-on-map-panel-${id}`

  useEffect(() => {
    const query = window.matchMedia('(prefers-color-scheme: dark)')
    const sync = () => setIsDark(query.matches)
    query.addEventListener('change', sync)
    return () => query.removeEventListener('change', sync)
  }, [])

  useEffect(() => {
    const onResize = () => {
      const width = window.innerWidth
      const height = window.innerHeight
      setViewport({ width, height })
      const raw = basePositionRef.current
      const next =
        width < 640 && (raw.x + BALL_SIZE > width || raw.y + BALL_SIZE > height)
          ? defaultPosition(width, height)
          : clampPosition(raw, width, height)
      basePositionRef.current = next
      x.set(next.x)
      y.set(next.y)
    }
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [x, y])

  useEffect(() => {
    const previous = previousFocusRef.current
    previousFocusRef.current = focusName
    if (!focusName || previous === focusName) return
    setPulse(true)
    const timer = window.setTimeout(() => setPulse(false), 700)
    return () => window.clearTimeout(timer)
  }, [focusName])

  useEffect(() => {
    if (isOpen) return
    setTarget(null)
    setResolveState('idle')
  }, [focusName, sessionKey, isOpen])

  useEffect(() => {
    const invalidate = () => {
      setConfig(null)
      setTarget(null)
      setResolveState('idle')
      setConfigRevision(value => value + 1)
    }
    window.addEventListener(AMAP_CONFIG_CHANGED, invalidate)
    return () => window.removeEventListener(AMAP_CONFIG_CHANGED, invalidate)
  }, [])

  useEffect(() => {
    setConfig(null)
    setTarget(null)
    if (!isOpen) return
    let disposed = false
    setResolveState('loading')
    void getAMapBrowserConfig().then(next => {
      if (disposed) return
      if (!next.key) { setResolveState('unconfigured'); return }
      setConfig({ ownerId: user!.user_id, value: next })
    }, () => { if (!disposed) setResolveState('error') })
    return () => { disposed = true }
  }, [configRevision, focusName, isOpen, sessionKey, user?.user_id])

  const close = useCallback(() => {
    setIsOpen(false)
    requestAnimationFrame(() => ballRef.current?.focus())
  }, [])

  useEffect(() => {
    if (!isOpen) return
    const frame = requestAnimationFrame(() => closeRef.current?.focus())
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      event.stopPropagation()
      close()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener('keydown', onKeyDown)
    }
  }, [close, isOpen])

  const openMap = useCallback(() => {
    snapControlsRef.current.forEach(control => control.stop())
    setCardPosition(basePositionRef.current)
    setIsOpen(true)
  }, [])

  const handlePointerDown = (event: ReactPointerEvent<HTMLButtonElement>) => {
    if (event.button !== 0) return
    snapControlsRef.current.forEach(control => control.stop())
    event.currentTarget.setPointerCapture(event.pointerId)
    draggedRef.current = false
    const base = basePositionRef.current
    dragRef.current = {
      pointerId: event.pointerId,
      offsetX: event.clientX - base.x,
      offsetY: event.clientY - base.y,
      startX: event.clientX,
      startY: event.clientY,
      moved: false,
    }
    setIsDragging(true)
  }

  const handlePointerMove = (event: ReactPointerEvent<HTMLButtonElement>) => {
    const drag = dragRef.current
    if (!drag || drag.pointerId !== event.pointerId) return
    if (!drag.moved) {
      if (Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) < 5) return
      drag.moved = true
      draggedRef.current = true
    }

    const base = clampPosition(
      { x: event.clientX - drag.offsetX, y: event.clientY - drag.offsetY },
      viewport.width,
      viewport.height,
    )
    basePositionRef.current = base
    const pull = magnetPull(base, viewport.width, viewport.height)
    x.set(base.x + pull.dx)
    y.set(base.y + pull.dy)
  }

  const finishDrag = (event: ReactPointerEvent<HTMLButtonElement>) => {
    const drag = dragRef.current
    if (!drag || drag.pointerId !== event.pointerId) return
    dragRef.current = null
    setIsDragging(false)
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
    if (!drag.moved) return

    const snapped = snapToEdge(basePositionRef.current, viewport.width, viewport.height)
    basePositionRef.current = snapped
    try {
      window.localStorage.setItem(POSITION_STORAGE_KEY, JSON.stringify(snapped))
    } catch {
      // 本地存储不可用时忽略
    }

    const transition = reduced
      ? { duration: 0 }
      : { type: 'spring' as const, stiffness: 260, damping: 26 }
    snapControlsRef.current = [animate(x, snapped.x, transition), animate(y, snapped.y, transition)]

    // 拖拽后若没有触发 click，下一次键盘激活不应被误吞。
    window.setTimeout(() => {
      draggedRef.current = false
    }, 0)
  }

  const handleClick = () => {
    if (draggedRef.current) {
      draggedRef.current = false
      return
    }
    openMap()
  }

  const cardSize = Math.min(380, viewport.width - 24, viewport.height - 24)
  const cardLeft = clamp(
    cardPosition.x + BALL_SIZE / 2 - cardSize / 2,
    12,
    Math.max(12, viewport.width - cardSize - 12),
  )
  const cardMaxTop = Math.max(12, viewport.height - cardSize - 12)
  const cardTop = clamp(
    cardPosition.y + BALL_SIZE / 2 - cardSize / 2,
    Math.min(72, cardMaxTop),
    cardMaxTop,
  )
  // 不用共享 layoutId：圆球由 motion values 定位，layout 投影会把 translate 再叠加一次，
  // 收起时球会先飞到屏幕外。改为显式计算“球心 → 卡片中心”的位移。
  const morphFrom = {
    x: cardPosition.x + BALL_SIZE / 2 - (cardLeft + cardSize / 2),
    y: cardPosition.y + BALL_SIZE / 2 - (cardTop + cardSize / 2),
  }
  const title = focusName ? `查看 ${focusName} 地图` : '查看当前位置'
  const chipLabel = target?.name ?? focusName ?? '当前位置'
  const loadingLabel = focusName ? `正在定位 ${focusName}…` : '正在获取当前位置…'
  const fallbackDetail =
    resolveState === 'unconfigured'
      ? '在设置 → MCP 接入 → 高德地图中配置'
      : resolveState === 'error'
        ? '地图加载失败，请检查配置后重试'
        : ''

  return (
    <AnimatePresence initial={false}>
      {isOpen ? (
        <motion.div
          key="map"
          id={panelId}
          role="dialog"
          aria-label={`${chipLabel}地图`}
          className="fixed z-[90] overflow-hidden rounded-2xl border border-border bg-bg-card shadow-2xl"
          style={{
            left: cardLeft,
            top: cardTop,
            width: cardSize,
            height: cardSize,
            transformOrigin: 'center',
          }}
          initial={{ opacity: 0, scale: 0.18, x: morphFrom.x, y: morphFrom.y }}
          animate={{ opacity: 1, scale: 1, x: 0, y: 0 }}
          exit={{ opacity: 0, scale: 0.18, x: morphFrom.x, y: morphFrom.y }}
          transition={reduced ? { duration: 0 } : { type: 'spring' as const, stiffness: 380, damping: 32, mass: 0.8 }}
        >
          {config && <MapCanvas key={`${user?.user_id}:${configRevision}:${sessionKey}:${focusName}:${isDark}`}
            config={config} focusName={focusName} isDark={isDark} onTarget={setTarget} onState={setResolveState} onClose={close} />}
          {resolveState === 'loading' || resolveState === 'idle' ? (
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 bg-bg-card">
              <LoaderCircle className="h-7 w-7 animate-spin text-text-muted" aria-hidden="true" />
              <span className="text-xs text-text-secondary">{loadingLabel}</span>
            </div>
          ) : resolveState !== 'ready' ? (
            <div
              className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2.5 bg-bg-card px-8 text-center"
              style={{ backgroundImage: FALLBACK_BG }}
            >
              <span className="grid h-12 w-12 place-items-center rounded-full bg-primary-light/10 text-primary-light">
                <MapPin className="h-6 w-6" aria-hidden="true" />
              </span>
              <span className="text-base font-semibold text-text-primary">{resolveState === 'unconfigured' ? '请配置高德地图 API' : '地图暂时不可用'}</span>
              <span className="text-xs leading-relaxed text-text-secondary">{fallbackDetail}</span>
            </div>
          ) : null}
          <span className="absolute left-3 top-3 z-20 max-w-[calc(100%-56px)] truncate rounded-full border border-border/70 bg-bg-card/90 px-3 py-1.5 text-xs font-medium text-text-primary shadow-sm backdrop-blur">
            {chipLabel}
          </span>
          <button
            ref={closeRef}
            type="button"
            aria-label="关闭地图"
            onClick={close}
            className="absolute right-3 top-3 z-30 grid h-9 w-9 place-items-center rounded-full border border-border bg-bg-card/95 text-text-secondary shadow-md backdrop-blur transition-colors hover:text-text-primary"
          >
            <X className="h-4 w-4" strokeWidth={2.5} aria-hidden="true" />
          </button>
        </motion.div>
      ) : (
        <motion.button
          key="ball"
          ref={ballRef}
          type="button"
          aria-haspopup="dialog"
          aria-label={title}
          title={title}
          className={[
            'group fixed left-0 top-0 z-[90] grid cursor-grab select-none place-items-center rounded-full text-white/80 shadow-[0_10px_30px_rgba(0,0,0,0.22)] backdrop-blur-[10px] touch-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary-light focus-visible:ring-offset-2',
            isDragging ? 'cursor-grabbing' : '',
            className ?? '',
          ]
            .filter(Boolean)
            .join(' ')}
          style={{ x, y, width: BALL_SIZE, height: BALL_SIZE }}
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.8 }}
          transition={reduced ? { duration: 0 } : { type: 'spring' as const, stiffness: 380, damping: 28 }}
          whileHover={reduced || isDragging ? undefined : { scale: 1.05 }}
          whileTap={reduced ? undefined : { scale: 0.95 }}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={finishDrag}
          onPointerCancel={finishDrag}
          onClick={handleClick}
        >
          <span
            aria-hidden="true"
            className={`absolute inset-0 rounded-full transition-colors duration-150 ${
              isDragging ? 'bg-black/70' : 'bg-black/50 group-hover:bg-black/60'
            }`}
          />
          {pulse && (
            <span className="pointer-events-none absolute inset-0 rounded-full ring-2 ring-white/50 motion-safe:animate-ping motion-reduce:hidden" />
          )}
          <MapIcon className="relative z-10 h-[22px] w-[22px] drop-shadow-[0_1px_2px_rgba(0,0,0,0.35)]" aria-hidden="true" />
        </motion.button>
      )}
    </AnimatePresence>
  )
}

export default ViewOnMap
