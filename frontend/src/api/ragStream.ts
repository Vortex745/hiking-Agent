export interface RagSourceDocument {
  title: string
  source: string
  content: string
  chunks?: number
  token?: string
  docType?: string
}

export interface RagSearchSummary {
  searchedCount: number
  matchedChunks: number
  documents: RagSourceDocument[]
}

export interface RagStreamState {
  content: string
  processSteps: string[]
  searchSummary: RagSearchSummary | null
}

export interface RagStreamEventLike {
  type: string
  content: string
  metadata?: Record<string, unknown>
}

export function createEmptyRagStreamState(): RagStreamState {
  return {
    content: '',
    processSteps: [],
    searchSummary: null,
  }
}

function toNumber(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback
}

function toText(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value : fallback
}

function normalizeDocuments(value: unknown): RagSourceDocument[] {
  if (!Array.isArray(value)) return []

  return value
    .filter((item): item is Record<string, unknown> => typeof item === 'object' && item !== null)
    .map(item => ({
      title: toText(item.title, '未命名文档'),
      source: toText(item.source, 'unknown'),
      content: toText(item.content),
      chunks: toNumber(item.chunks, 0),
      token: toText(item.token),
      docType: toText(item.doc_type),
    }))
}

function normalizeSearchSummary(metadata: Record<string, unknown> | undefined): RagSearchSummary | null {
  if (!metadata) return null

  return {
    searchedCount: toNumber(metadata.searched_count),
    matchedChunks: toNumber(metadata.matched_chunks),
    documents: normalizeDocuments(metadata.documents),
  }
}

export function applyRagStreamEvent(
  state: RagStreamState,
  event: RagStreamEventLike,
): RagStreamState {
  if (event.type === 'text') {
    return { ...state, content: state.content + event.content }
  }

  if (event.type === 'process' || event.type === 'thought') {
    const step = event.content.trim()
    if (!step) return state
    return { ...state, processSteps: [...state.processSteps, step] }
  }

  if (event.type === 'documents') {
    return {
      ...state,
      searchSummary: normalizeSearchSummary(event.metadata),
    }
  }

  return state
}

