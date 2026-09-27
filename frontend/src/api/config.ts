const configuredApiBase = import.meta.env?.VITE_API_BASE_URL?.trim().replace(/\/+$/, '')

// models 端点仍由 AI service 的 /api/v1 前缀提供。
const API_BASE = configuredApiBase || '/api/v1'

// Chat v2 端点位于 /api/v2，由长驻 Docker gateway 提供服务。
// v1 chat 已移除，v2 是唯一 chat 链路。
const V2_BASE = configuredApiBase
  ? configuredApiBase.replace(/\/api\/v1\/?$/, '/api/v2')
  : '/api/v2'

export const API = {
  modelsFetch: `${API_BASE}/models/fetch`,

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
