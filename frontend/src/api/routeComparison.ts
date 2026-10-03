export type RouteFactKey = 'distance_km' | 'ascent_m' | 'duration_hours' | 'weather' | 'map' | 'trail_status' | 'risk'
export interface RouteSource { id: string; tool: string; url: string | null; excerpt: string; observed_at: string }
export interface RouteFact { value: string | number; source_id: string; source_field: string }
export interface RouteAssessment {
  candidate: { id: string; name: string; facts: Partial<Record<RouteFactKey, RouteFact | null>>; sources: RouteSource[] }
  evidence_status: 'complete' | 'partial' | 'empty'
  risk_status: 'clear' | 'caution' | 'confirmation_required' | 'no_go'
  reasons: string[]
  required_action: string | null
}
export interface RouteComparison {
  comparison_id: string
  status: 'recommended' | 'confirmation_required' | 'insufficient_evidence' | 'no_go' | 'provider_unavailable'
  decision: { provider: string; model: string | null; status: string; reason: string | null; confidence: { kind: 'heuristic' | 'calibrated' | 'unavailable'; value: number | null } }
  candidates: RouteAssessment[]
  recommended_id: string | null
  alternatives: string[]
  explanation: string[]
  report_allowed: boolean
  generated_at: string
  data_mode: 'live' | 'offline'
}

const record = (value: unknown): value is Record<string, unknown> => Boolean(value && typeof value === 'object' && !Array.isArray(value))
const strings = (value: unknown): value is string[] => Array.isArray(value) && value.every(item => typeof item === 'string')

export function readRouteComparison(value: unknown): RouteComparison | null {
  if (!record(value) || typeof value.comparison_id !== 'string' ||
      !['recommended', 'confirmation_required', 'insufficient_evidence', 'no_go', 'provider_unavailable'].includes(String(value.status)) ||
      !record(value.decision) || !record(value.decision.confidence) ||
      typeof value.decision.provider !== 'string' ||
      !['heuristic', 'calibrated', 'unavailable'].includes(String(value.decision.confidence.kind)) ||
      !Array.isArray(value.candidates) || value.candidates.length > 5 ||
      !strings(value.alternatives) || !strings(value.explanation)) return null
  for (const assessment of value.candidates) {
    if (!record(assessment) || !record(assessment.candidate) ||
        typeof assessment.candidate.id !== 'string' || typeof assessment.candidate.name !== 'string' ||
        !record(assessment.candidate.facts) || !Array.isArray(assessment.candidate.sources) ||
        !strings(assessment.reasons) ||
        !['clear', 'caution', 'confirmation_required', 'no_go'].includes(String(assessment.risk_status)) ||
        !['complete', 'partial', 'empty'].includes(String(assessment.evidence_status))) return null
    for (const fact of Object.values(assessment.candidate.facts)) {
      if (fact !== null && (!record(fact) || !['string', 'number'].includes(typeof fact.value) || typeof fact.source_id !== 'string')) return null
    }
    for (const source of assessment.candidate.sources) {
      if (!record(source) || typeof source.id !== 'string' || typeof source.tool !== 'string' ||
          typeof source.excerpt !== 'string' || typeof source.observed_at !== 'string' ||
          (source.url !== null && typeof source.url !== 'string')) return null
    }
  }
  return value as unknown as RouteComparison
}

export function safeSourceUrl(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined
  try {
    const url = new URL(value)
    return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password ? url.href : undefined
  } catch { return undefined }
}
