import { readRouteComparison, safeSourceUrl, type RouteFactKey, type RouteSelection } from '../api/routeComparison'

const dimensions: Array<[RouteFactKey, string, string]> = [
  ['distance_km', '距离', 'km'], ['ascent_m', '爬升', 'm'], ['duration_hours', '耗时', '小时'],
  ['weather', '天气', ''], ['map', '地图', ''], ['trail_status', '开放状态', ''], ['risk', '风险资料', ''],
]
const statuses = { recommended: '已核验推荐', confirmation_required: '需要确认', insufficient_evidence: '资料不足', no_go: '禁止出行', provider_unavailable: '决策服务不可用' }
const risks = { clear: '风险资料齐全', caution: '谨慎 · 需要确认', confirmation_required: '安全资料待确认', no_go: 'No-go · 禁止推荐' }
const evidence = { complete: '资料齐全', partial: '部分资料', empty: '无可核验资料' }
const values: Record<string, string> = { open: '开放', closed: '封闭', low: '低风险', moderate: '中等风险', high: '高风险' }

export function RouteComparisonCard({ value, onExport, onSelect, busy = false }: {
  value: unknown; onExport?: (comparisonId: string) => void; busy?: boolean
  /**
   * Ticket 03 AC1: send a route into the plan draft. The card is the only place
   * a selection can be made — it shows the verified facts and their sources, so
   * it is also the only place that can say which candidate the comparison
   * actually offered. The action carries two ids: a draft that reuses a
   * comparison stores a reference, never a copy of these facts.
   */
  onSelect?: (selection: RouteSelection) => void
}) {
  const result = readRouteComparison(value)
  if (!result) return null
  const validRecommendation = result.status === 'recommended' && result.decision.status === 'ok'
  return (
    <section aria-label="路线比较" className="my-4 rounded-xl border border-border bg-bg-card p-4 text-text-primary">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-base font-semibold">路线比较 · {statuses[result.status]}</h3>
          <p className="mt-1 text-xs text-text-secondary">
            {result.decision.provider === 'llm' ? '普通 LLM' : result.decision.provider === 'external' ? '外部决策模型' : '离线确定性决策'}
            {' · '}{result.decision.model || '尚未调用'}
            {' · '}{result.decision.confidence.kind === 'calibrated' ? '校准 confidence' : result.decision.confidence.kind === 'heuristic' ? '启发式 confidence' : 'confidence 不可用'}
            {typeof result.decision.confidence.value === 'number' && ` ${Math.round(result.decision.confidence.value * 100)}%`}
          </p>
          <p className="mt-1 text-xs text-text-muted">confidence 描述模型分布，不代表路线安全保证。</p>
        </div>
        {validRecommendation && result.report_allowed && (
          <button type="button" disabled={busy || !onExport} onClick={() => onExport?.(result.comparison_id)}
            className="rounded-lg border border-border px-3 py-2 text-sm hover:bg-bg-body disabled:opacity-50">导出带来源 PDF</button>
        )}
      </div>
      {result.data_mode === 'offline' && <p className="mt-3 rounded-lg bg-amber-500/10 p-2 text-sm">离线演示 · 合成数据，不可用于真实出行。</p>}
      {result.decision.reason && <p className="mt-3 text-sm text-text-secondary">决策状态：{result.decision.reason}</p>}
      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        {result.candidates.map(assessment => {
          const candidate = assessment.candidate
          const recommended = validRecommendation && result.recommended_id === candidate.id && assessment.risk_status === 'clear' && assessment.evidence_status === 'complete'
          const alternate = validRecommendation && result.alternatives.includes(candidate.id)
          // AC1/AC4: only what the comparison offered is selectable. A no-go, an
          // insufficient-evidence or an off-ranking candidate gets no affordance,
          // because choosing it could only ever produce an unverified draft.
          const selectable = recommended || alternate
          const selection: RouteSelection = { comparison_id: result.comparison_id, candidate_id: candidate.id }
          return <article key={candidate.id} className={`rounded-lg border p-3 ${recommended ? 'border-primary' : 'border-border'}`}>
            <h4 className="font-medium">{candidate.name}{recommended ? ' · 推荐' : alternate ? ' · 备选' : ''}</h4>
            <p className={`mt-1 text-xs ${assessment.risk_status === 'no_go' ? 'text-red-600 dark:text-red-400' : 'text-text-secondary'}`}>
              {evidence[assessment.evidence_status]} · {risks[assessment.risk_status]}
            </p>
            <dl className="mt-3 space-y-2 text-sm">
              {dimensions.map(([key, label, unit]) => {
                const fact = candidate.facts[key]
                const source = fact && candidate.sources.find(item => item.id === fact.source_id)
                return <div key={key}>
                  <dt className="text-xs text-text-muted">{label}</dt>
                  <dd>{fact && source ? <details>
                    <summary className="cursor-pointer">{key === 'map' ? '查看地图及来源' : `${values[String(fact.value)] || fact.value} ${unit}`}</summary>
                    <p className="mt-1 break-words text-xs text-text-secondary">{source.excerpt}</p>
                    {safeSourceUrl(source.url || (key === 'map' ? fact.value : undefined)) && <a
                      className="text-xs underline" target="_blank" rel="noopener noreferrer"
                      href={safeSourceUrl(source.url || (key === 'map' ? fact.value : undefined))}>打开来源</a>}
                    <p className="text-xs text-text-muted">{source.tool} · {source.observed_at}</p>
                  </details> : <span className="text-text-secondary">资料不足</span>}</dd>
                </div>
              })}
            </dl>
            {assessment.reasons.length > 0 && <ul className="mt-3 space-y-1 text-xs text-text-secondary">{assessment.reasons.map((reason, i) => <li key={i}>{reason}</li>)}</ul>}
            {assessment.required_action && <p className="mt-2 text-sm font-medium">{assessment.required_action}</p>}
            {selectable && <button type="button" disabled={busy || !onSelect}
              onClick={() => onSelect?.(selection)}
              className="mt-3 rounded-lg border border-border px-3 py-2 text-sm hover:bg-bg-body disabled:opacity-50 dark:border-white/10">
              {recommended ? '选用该路线（推荐）' : '选用该路线（备选）'}
            </button>}
            {!selectable && onSelect && <p className="mt-3 text-xs text-text-muted">该路线不在推荐或备选之列，无法带入计划书草稿。</p>}
          </article>
        })}
      </div>
      {result.required_action && <p className="mt-3 rounded-lg bg-amber-500/10 p-2 text-sm">{result.required_action}</p>}
      <details className="mt-3 text-sm"><summary className="cursor-pointer">比较依据、资料缺口与用户安排</summary>
        <ul className="mt-2 space-y-1 break-words text-text-secondary">{result.explanation.map((line, i) => <li key={i}>{line}</li>)}</ul>
        {result.ranking.length > 0 && <p className="mt-2 break-words text-xs text-text-muted">
          排序：{result.ranking.map(id => result.candidates.find(a => a.candidate.id === id)?.candidate.name || id).join(' → ')}
        </p>}
        {result.candidates.some(assessment => assessment.candidate.gaps.length > 0) && (
          <div className="mt-2 space-y-2">
            {result.candidates.map(assessment => assessment.candidate.gaps.length > 0 && (
              <div key={assessment.candidate.id}>
                <p className="text-xs text-text-muted">{assessment.candidate.name} · 资料缺口</p>
                <ul className="space-y-1 break-words text-xs text-text-secondary">
                  {assessment.candidate.gaps.map((gap, i) => <li key={i}>{gap}</li>)}
                </ul>
              </div>
            ))}
          </div>
        )}
        <p className="mt-2 text-xs text-text-muted">
          用户安排（待确认，非核验事实）：活动日期 {result.plan_details.activity_date || '待确认'}
          ，集合地点 {result.plan_details.meeting_point || '待确认'}
          ，参与人数 {result.plan_details.participants ?? '待确认'}
          ，交通 {result.plan_details.transport || '待确认'}
        </p>
      </details>
    </section>
  )
}
