const configuredApiBase = import.meta.env?.VITE_API_BASE_URL?.trim().replace(/\/+$/, '')

const API_BASE = configuredApiBase || '/api/v1'

// Ticket 08: v2 endpoints live under /api/v2 and are served by the long-lived
// Docker gateway. The base is derived independently so v1 and v2 can coexist
// during the canary without one affecting the other's routing.
const V2_BASE = configuredApiBase
  ? configuredApiBase.replace(/\/api\/v1\/?$/, '/api/v2')
  : '/api/v2'

export const API = {
  chatRuntime: (conversationId: string) =>
    `/api/chat/runtime/${encodeURIComponent(conversationId)}`,
  chatSSE: `${API_BASE}/chat/sse`,
  chatSync: `${API_BASE}/chat/sync`,
  chatHealth: `${API_BASE}/chat/health`,
  chatConfirm: `${API_BASE}/chat/confirm`,
  chatPending: (chatId: string) => `${API_BASE}/chat/pending/${encodeURIComponent(chatId)}`,
  chatHistory: (chatId: string) => `${API_BASE}/chat/history/${encodeURIComponent(chatId)}`,
  chatHistoryDelete: (chatId: string) => `${API_BASE}/chat/history/${encodeURIComponent(chatId)}`,

  modelsFetch: `${API_BASE}/models/fetch`,

  ragQuery: `${API_BASE}/rag/query`,
  ragUpload: `${API_BASE}/rag/upload`,
  ragHealth: `${API_BASE}/rag/health`,
  ragDocuments: `${API_BASE}/rag/documents`,
  ragHistory: (chatId: string) => `${API_BASE}/rag/history/${encodeURIComponent(chatId)}`,
  ragHistoryDelete: (chatId: string) => `${API_BASE}/rag/history/${encodeURIComponent(chatId)}`,

  // Ticket 08: Chat v2 endpoints. The SSE endpoint is the only streaming
  // surface; snapshot/cancel/events support refresh recovery and cancel.
  chatV2SSE: `${V2_BASE}/chat/sse`,
  chatV2Snapshot: (runId: string) => `${V2_BASE}/chat/snapshot/${encodeURIComponent(runId)}`,
  chatV2Cancel: (runId: string) => `${V2_BASE}/chat/cancel/${encodeURIComponent(runId)}`,
  chatV2Events: (runId: string, fromSeq = 0) =>
    `${V2_BASE}/chat/events/${encodeURIComponent(runId)}?from_seq=${fromSeq}`,

  // Ticket 11: durable approvals + resume. The decision endpoint records
  // the user's verdict; the resume endpoint re-opens the SSE stream so
  // the same run continues with the same checkpoint and sandbox.
  chatV2ApprovalDecision: (approvalId: string) =>
    `${V2_BASE}/approvals/${encodeURIComponent(approvalId)}/decision`,
  chatV2Resume: (runId: string, fromSeq = 0) =>
    `${V2_BASE}/chat/resume/${encodeURIComponent(runId)}?from_seq=${fromSeq}`,
  chatV2MemorySettings: `${V2_BASE}/memory/settings`,
  chatV2Memories: (limit = 100, offset = 0) =>
    `${V2_BASE}/memories?limit=${limit}&offset=${offset}`,
  chatV2Memory: (memoryId: string) =>
    `${V2_BASE}/memories/${encodeURIComponent(memoryId)}`,
  chatV2MemoryClear: `${V2_BASE}/memories`,
  chatV2MemoryExport: `${V2_BASE}/memories/export`,
  chatV2MemoryCandidateDecision: (candidateId: string) =>
    `${V2_BASE}/memory/candidates/${encodeURIComponent(candidateId)}/decision`,
  chatV2Artifact: (artifactId: string) =>
    `${V2_BASE}/artifacts/${encodeURIComponent(artifactId)}`,
} as const
