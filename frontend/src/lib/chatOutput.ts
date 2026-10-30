/**
 * Legacy MCP output sanitizer for the chat transcript.
 *
 * Older backend turns appended `MCP 返回：` followed by pretty-printed JSON
 * envelopes. Those messages are already persisted, so fixing the writer only
 * cleans new turns. This display-time helper strips the JSON blocks (and
 * extracts their image URLs) without touching the stored history.
 */

const MARKER = 'MCP 返回：'
const MAX_IMAGES = 12
const IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.avif', '.bmp']
const IMAGE_KEY_HINTS = new Set([
  'preview_url',
  'image_url',
  'image_urls',
  'thumbnail_url',
  'thumb_url',
  'cover_url',
  'download_url',
])

interface LegacyImage {
  url: string
  alt: string
}

function isHttpUrl(value: string): boolean {
  try {
    const parsed = new URL(value)
    return parsed.protocol === 'http:' || parsed.protocol === 'https:'
  } catch {
    return false
  }
}

function looksLikeImageUrl(value: string, key: string | null): boolean {
  if (!isHttpUrl(value)) return false
  let path = ''
  try {
    path = new URL(value).pathname.toLowerCase()
  } catch {
    return false
  }
  if (IMAGE_EXTENSIONS.some(ext => path.endsWith(ext))) return true
  return key !== null && IMAGE_KEY_HINTS.has(key)
}

function cleanAlt(title: unknown, index: number): string {
  if (typeof title === 'string') {
    const cleaned = title.replace(/\[/g, '').replace(/\]/g, '').replace(/\n/g, ' ').trim()
    if (cleaned) return cleaned.slice(0, 60)
  }
  return `MCP 图片 ${index}`
}

function collectImages(value: unknown, key: string | null, title: unknown, found: LegacyImage[], seen: Set<string>): void {
  if (found.length >= MAX_IMAGES) return
  if (Array.isArray(value)) {
    for (const item of value) collectImages(item, key, title, found, seen)
    return
  }
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>
    const localTitle = record.title ?? record.description ?? title
    for (const [childKey, child] of Object.entries(record)) {
      collectImages(child, childKey, localTitle, found, seen)
    }
    return
  }
  if (typeof value !== 'string' || !looksLikeImageUrl(value, key)) return
  if (seen.has(value)) return
  seen.add(value)
  found.push({ url: value, alt: cleanAlt(title, found.length + 1) })
}

export function splitLegacyMcpOutput(raw: string): string {
  if (!raw || !raw.includes(MARKER)) return raw
  const markerIndex = raw.indexOf(MARKER)
  const prefix = raw.slice(0, markerIndex).trimEnd()
  const suffix = raw.slice(markerIndex + MARKER.length)
  const chunks = suffix.split(/\n{2,}/).map(chunk => chunk.trim()).filter(Boolean)
  if (chunks.length === 0) return raw

  let parsedJson = false
  const textChunks: string[] = []
  const images: LegacyImage[] = []
  const seen = new Set<string>()
  for (const chunk of chunks) {
    let payload: unknown
    try {
      payload = JSON.parse(chunk)
    } catch {
      textChunks.push(chunk)
      continue
    }
    parsedJson = true
    collectImages(payload, null, null, images, seen)
  }

  // The marker can appear in ordinary assistant prose. Only rewrite when a
  // JSON block was actually found after it; otherwise leave the text intact.
  if (!parsedJson) return raw

  const parts: string[] = []
  if (prefix) parts.push(prefix)
  if (textChunks.length) parts.push(`${MARKER}\n${textChunks.join('\n\n')}`)
  if (images.length) {
    const lines = images.map(image => `![${image.alt}](${image.url})`)
    parts.push(lines.join('\n'))
  }
  return parts.join('\n\n')
}
