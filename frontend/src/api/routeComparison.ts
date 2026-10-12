export type RouteFactKey = 'distance_km' | 'ascent_m' | 'duration_hours' | 'weather' | 'map' | 'trail_status' | 'risk'
export interface RouteSource { id: string; tool: string; url: string | null; excerpt: string; observed_at: string }
export interface RouteFact { value: string | number; source_id: string; source_field: string }
/**
 * Ticket 03: the user's own arrangements travel with a comparison result
 * (`PlanDetails` on the server). They are a DIFFERENT kind of claim from a
 * verified fact — an empty string here means 待确认, never 0 or 未知.
 */
export interface RoutePlanStage { time: string; activity: string }
export interface RouteBudgetItem { item: string; amount_yuan: number }
export interface RoutePlanDetails {
  activity_date: string
  meeting_time: string
  meeting_point: string
  participants: number | null
  leader: string
  transport: string
  accommodation: string
  emergency_contact: string
  retreat_plan: string
  itinerary: RoutePlanStage[]
  budget: RouteBudgetItem[]
}
export interface RouteAssessment {
  candidate: {
    id: string
    name: string
    facts: Partial<Record<RouteFactKey, RouteFact | null>>
    sources: RouteSource[]
    /** Evidence gaps the deterministic gate recorded for this candidate. */
    gaps: string[]
  }
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
  /** The gate-passed candidates in decision order: index 0 is the recommendation. */
  ranking: string[]
  recommended_id: string | null
  alternatives: string[]
  explanation: string[]
  /** What the user must do before this comparison can go further. */
  required_action: string | null
  report_allowed: boolean
  generated_at: string
  data_mode: 'live' | 'offline'
  plan_details: RoutePlanDetails
}

/** The pair a plan draft points at: a selection is ids only, never a fact copy. */
export interface RouteSelection { comparison_id: string; candidate_id: string }

const record = (value: unknown): value is Record<string, unknown> => Boolean(value && typeof value === 'object' && !Array.isArray(value))
const strings = (value: unknown): value is string[] => Array.isArray(value) && value.every(item => typeof item === 'string')
const text = (value: unknown): string => typeof value === 'string' ? value : ''

/**
 * Read one candidate's arrangements. Absent or malformed entries collapse to
 * the empty/``null`` shape the server itself uses for an uncollected field —
 * this reader has no parameter that accepts a value, so it cannot fill one in.
 */
function readPlanDetails(value: unknown): RoutePlanDetails {
  const raw = record(value) ? value : {}
  const participants = raw.participants
  const itinerary = Array.isArray(raw.itinerary) ? raw.itinerary : []
  const budget = Array.isArray(raw.budget) ? raw.budget : []
  return {
    activity_date: text(raw.activity_date),
    meeting_time: text(raw.meeting_time),
    meeting_point: text(raw.meeting_point),
    participants: typeof participants === 'number' && Number.isFinite(participants) ? participants : null,
    leader: text(raw.leader),
    transport: text(raw.transport),
    accommodation: text(raw.accommodation),
    emergency_contact: text(raw.emergency_contact),
    retreat_plan: text(raw.retreat_plan),
    itinerary: itinerary.filter(record).map(stage => ({ time: text(stage.time), activity: text(stage.activity) })),
    budget: budget.filter(record)
      .filter(item => typeof item.item === 'string' && typeof item.amount_yuan === 'number')
      .map(item => ({ item: item.item as string, amount_yuan: item.amount_yuan as number })),
  }
}

export function readRouteComparison(value: unknown): RouteComparison | null {
  if (!record(value) || typeof value.comparison_id !== 'string' ||
      !['recommended', 'confirmation_required', 'insufficient_evidence', 'no_go', 'provider_unavailable'].includes(String(value.status)) ||
      !record(value.decision) || !record(value.decision.confidence) ||
      typeof value.decision.provider !== 'string' ||
      !['heuristic', 'calibrated', 'unavailable'].includes(String(value.decision.confidence.kind)) ||
      !Array.isArray(value.candidates) || value.candidates.length > 5 ||
      !strings(value.alternatives) || !strings(value.explanation) ||
      // Optional-to-older-payloads, but never malformed-and-accepted.
      (value.ranking !== undefined && !strings(value.ranking)) ||
      (value.required_action !== undefined && value.required_action !== null &&
        typeof value.required_action !== 'string')) return null
  const assessments: RouteAssessment[] = []
  for (const assessment of value.candidates) {
    if (!record(assessment) || !record(assessment.candidate) ||
        typeof assessment.candidate.id !== 'string' || typeof assessment.candidate.name !== 'string' ||
        !record(assessment.candidate.facts) || !Array.isArray(assessment.candidate.sources) ||
        !strings(assessment.reasons) ||
        (assessment.candidate.gaps !== undefined && !strings(assessment.candidate.gaps)) ||
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
    // ``gaps`` is normalised rather than cast so a candidate rendered before
    // ticket 03 still shows an honest empty list instead of ``undefined``.
    assessments.push({
      ...(assessment as unknown as RouteAssessment),
      candidate: {
        ...(assessment.candidate as unknown as RouteAssessment['candidate']),
        gaps: strings(assessment.candidate.gaps) ? assessment.candidate.gaps : [],
      },
    })
  }
  return {
    ...(value as unknown as RouteComparison),
    candidates: assessments,
    ranking: strings(value.ranking) ? value.ranking : [],
    required_action: typeof value.required_action === 'string' ? value.required_action : null,
    plan_details: readPlanDetails(value.plan_details),
  }
}

export function safeSourceUrl(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined
  try {
    const url = new URL(value)
    return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password ? url.href : undefined
  } catch { return undefined }
}
