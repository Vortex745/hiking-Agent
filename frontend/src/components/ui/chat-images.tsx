/**
 * Chat image display: images stay in the message flow and open in Viewer.js
 * for a modal zoom view. No stacking/fan effect.
 */
import { useCallback, useEffect, useRef, type MouseEvent as ReactMouseEvent } from 'react'
import Viewer from 'viewerjs'
import 'viewerjs/dist/viewer.css'

export type ChatImage = { src: string; alt: string }

const viewerOptions = (multi: boolean) => ({
  title: false,
  loop: false,
  keyboard: true,
  zIndex: 2100,
  // Images are lazy-loaded; copying `loading="lazy"` into the modal viewer
  // leaves its <img> unloaded, so the viewer inherits only src/alt.
  inheritedAttributes: [] as string[],
  navbar: multi ? ('small' as const) : (false as const),
  toolbar: {
    zoomIn: true,
    zoomOut: true,
    oneToOne: true,
    reset: true,
    rotateLeft: true,
    rotateRight: true,
    prev: multi,
    next: multi,
    play: false,
    flipHorizontal: false,
    flipVertical: false,
  },
})

export function ChatImages({ images }: { images: ChatImage[] }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const viewerRef = useRef<Viewer | null>(null)
  const viewerCountRef = useRef(0)
  const openerRef = useRef<HTMLButtonElement | null>(null)

  useEffect(() => {
    viewerRef.current?.update()
  }, [images.length])

  useEffect(() => () => {
    viewerRef.current?.destroy()
    viewerRef.current = null
  }, [])

  const openViewer = useCallback((index: number, event: ReactMouseEvent<HTMLButtonElement>) => {
    openerRef.current = event.currentTarget
    const node = containerRef.current
    if (!node) return
    let viewer = viewerRef.current
    if (viewer && viewerCountRef.current === images.length) {
      // Viewer.js binds its own delegated click handler on the container, so
      // later pointer clicks on an <img> already open the viewer.
      if (event.target instanceof HTMLImageElement) return
    } else {
      viewer?.destroy()
      viewer = new Viewer(node, {
        ...viewerOptions(images.length > 1),
        hidden: () => openerRef.current?.focus(),
      })
      viewerRef.current = viewer
      viewerCountRef.current = images.length
    }
    viewer.view(index)
  }, [images.length])

  return (
    <div
      ref={containerRef}
      className="my-2 flex max-w-full flex-col gap-2"
      data-chat-images={images.length > 1 ? 'multi' : 'single'}
      data-image-count={images.length}
    >
      {images.map((image, index) => (
        <button
          key={`${image.src}-${index}`}
          type="button"
          onClick={(event) => openViewer(index, event)}
          className="block w-fit max-w-full cursor-zoom-in overflow-hidden rounded-xl border border-black/5 shadow-sm transition-opacity hover:opacity-95 focus:outline-none focus-visible:ring-2 focus-visible:ring-[#0b57d0]/60 dark:border-white/5"
          aria-label={`放大图片：${image.alt}`}
          title="放大图片"
        >
          <img
            src={image.src}
            alt={image.alt}
            className="block max-h-[350px] max-w-full object-cover"
            loading="lazy"
            draggable={false}
          />
        </button>
      ))}
    </div>
  )
}
