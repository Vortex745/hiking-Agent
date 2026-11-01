// Structured plan entry, mobile form, refresh and account isolation.
const { chromium } = require('playwright')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

;(async () => {
  const output = process.env.SCREENSHOT_DIR || path.resolve(__dirname, '../../.screenshots/plan-workflow')
  fs.mkdirSync(output, { recursive: true })
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH })
  const page = await browser.newPage({ viewport: { width: 390, height: 812 } })
  let account = 'alice'
  const requests = []
  const errors = []
  page.on('pageerror', error => errors.push(String(error)))
  const json = (route, body) => route.fulfill({ contentType: 'application/json', body: JSON.stringify(body) })
  try {
    await page.route('**/api/**', async route => {
      const url = new URL(route.request().url())
      if (url.pathname === '/api/auth/me') return json(route, { user_id: account, login_id: account, created_at: '2026-11-02T00:00:00Z' })
      if (url.pathname === '/api/v2/memory/settings') return json(route, { subject_id: account, enabled: false })
      if (url.pathname.endsWith('/chat/sse')) {
        const body = route.request().postDataJSON()
        requests.push(body)
        const submitted = Boolean(body.plan_input)
        const draft = body.plan_input || { destination_or_route: '', activity_date: '', participants: null }
        const payload = {
          stage: submitted ? 'minimal_draft' : 'awaiting_input', draft_id: 'fixture-draft',
          conversation_id: body.conversation_id, source: 'structured_intent', intent: body.intent,
          summary: submitted ? `目的地：${draft.destination_or_route}` : '', plan_type: 'generic_draft',
          route_selection: null, missing_fields: [], rules: null, safety: null, draft,
        }
        const events = [['run_started', {}], ['plan.draft_state', payload], ['text', { content: payload.summary || '请填写计划书' }], ['done', {}]]
        const stream = events.map(([type, payload], index) => `data: ${JSON.stringify({
          version: '2', event_id: `${requests.length}-${index}`, sequence: index + 1,
          run_id: `run-${requests.length}`, request_id: body.request_id,
          conversation_id: body.conversation_id, timestamp: new Date().toISOString(), type, payload,
        })}\n\n`).join('')
        return route.fulfill({ contentType: 'text/event-stream', body: stream })
      }
      return json(route, { enabled: false })
    })
    await page.goto(process.env.BASE_URL || 'http://127.0.0.1:18101')
    await page.getByRole('button', { name: '生成行程', exact: true }).click()
    const dialog = page.getByRole('dialog', { name: '生成行程计划书' })
    await dialog.waitFor()
    assert.equal(requests[0].intent, 'plan_create')
    await dialog.locator('input[name="destination_or_route"]').fill('账号 A 的林间环线')
    await dialog.locator('input[name="activity_date"]').fill('2026-11-15')
    await dialog.locator('input[name="participants"]').fill('4')
    const panel = dialog.getByRole('document')
    const bounds = await panel.boundingBox()
    assert.ok(bounds.y >= 0 && bounds.y + bounds.height <= 813, 'mobile dialog stays within viewport')
    await dialog.getByRole('button', { name: '生成最小草稿' }).click()
    await dialog.waitFor({ state: 'hidden' })
    assert.equal(requests[1].intent, 'plan_create')
    assert.equal(requests[1].plan_input.participants, 4)
    await page.reload()
    await page.getByRole('button', { name: '恢复草稿', exact: true }).click()
    await dialog.waitFor()
    assert.equal(await dialog.locator('input[name="destination_or_route"]').inputValue(), '账号 A 的林间环线')
    await page.screenshot({ path: path.join(output, 'mobile-restored.png') })
    await page.keyboard.press('Escape')
    account = 'bob'
    await page.reload()
    await page.getByRole('button', { name: '生成行程', exact: true }).waitFor()
    assert.equal(await page.getByRole('button', { name: '恢复草稿', exact: true }).count(), 0)
    assert.equal(await page.getByText('账号 A 的林间环线', { exact: true }).count(), 0)
    const stored = await page.evaluate(() => Object.fromEntries(Object.entries(localStorage).filter(([key]) => key.includes('plan-drafts'))))
    assert.ok(stored['u:alice:ai-hiking-agent-plan-drafts'])
    assert.equal(stored['ai-hiking-agent-plan-drafts'], undefined)
    account = 'alice'
    await page.reload()
    await page.getByRole('button', { name: '恢复草稿', exact: true }).waitFor()
    assert.deepEqual(errors, [])
    console.log('PASS: structured entry/submit, mobile scrolling, refresh, A→B→A plan isolation')
  } finally {
    await browser.close()
  }
})().catch(error => { console.error(error); process.exitCode = 1 })
