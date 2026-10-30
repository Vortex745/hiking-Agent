// 账号级模型配置：来源标记与“恢复系统默认”浏览器回归。
// Run: BASE_URL=http://127.0.0.1:5174 CHROME_PATH=... node tests/llm-config-account.cjs
const { chromium } = require('playwright')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const screenshotDir = process.env.SCREENSHOT_DIR || path.resolve(__dirname, '../../.screenshots/jwt-auth')
const base = { base_url: 'https://llm.example/v1', api_key_masked: 'sk-a...0000', api_key_set: true }
let resets = 0

function snapshot(sources) {
  return {
    llm: { ...base, model: sources.llm === 'account' ? 'account-model' : 'system-model' },
    embedding: { ...base, model: 'emb-model', dimensions: 1024 },
    rerank: { ...base, model: 'rerank-model' },
    decision: { ...base, model: 'Kev-4b', mode: 'off' },
    updated_at: '2026-10-30T00:00:00Z',
    index_dimensions: 1024,
    sources,
  }
}

(async () => {
  fs.mkdirSync(screenshotDir, { recursive: true })
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH })
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, reducedMotion: 'reduce' })
    const errors = []
    page.on('pageerror', error => errors.push(String(error)))
    page.on('console', message => {
      if (message.type() === 'error' && !/Failed to load resource/.test(message.text())) errors.push(message.text())
    })

    await page.route('**/*', route => {
      const request = route.request()
      const pathname = new URL(request.url()).pathname
      if (pathname === '/api/auth/me') {
        return route.fulfill({ json: { user_id: 'user-alice', login_id: 'Abc1', created_at: '2026-10-30T00:00:00Z' } })
      }
      if (pathname === '/api/v1/llm-config' && request.method() === 'GET') {
        return route.fulfill({
          json: snapshot({ llm: 'account', embedding: 'system', rerank: 'system', decision: 'system' }),
        })
      }
      if (pathname === '/api/v1/llm-config' && request.method() === 'DELETE') {
        resets += 1
        return route.fulfill({
          json: { ...snapshot({ llm: 'system', embedding: 'system', rerank: 'system', decision: 'system' }), reindex_recommended: false },
        })
      }
      if (pathname === '/api/v2/mcp/servers') return route.fulfill({ json: { servers: [] } })
      if (pathname === '/api/v2/memory/settings') return route.fulfill({ json: { enabled: false } })
      if (pathname.startsWith('/api/')) return route.fulfill({ json: { enabled: false } })
      return route.continue()
    })
    await page.addInitScript(() => {
      navigator.geolocation.getCurrentPosition = (_success, failure) => failure?.({ code: 1, message: 'fixture' })
    })

    await page.goto(process.env.BASE_URL || 'http://127.0.0.1:5174', { waitUntil: 'domcontentloaded' })
    await page.getByRole('button', { name: '设置' }).click()
    await page.getByRole('menuitem', { name: '模型配置' }).click()

    const dialog = page.getByRole('dialog', { name: '模型配置' })
    await dialog.waitFor()
    await dialog.getByText('当前账号').waitFor()
    await dialog.getByText('Abc1').waitFor()
    await dialog.getByText('大模型配置 · 账号').waitFor()
    await dialog.getByText('Embedding 配置 · 系统').waitFor()
    assert.equal(await dialog.locator('.arc-card').count(), 4, 'must still show 4 config cards')
    await page.screenshot({ path: path.join(screenshotDir, '09-config-account.png'), animations: 'disabled' })

    await dialog.getByRole('button', { name: '恢复系统默认' }).click()
    await dialog.getByText('已恢复系统默认', { exact: true }).waitFor()
    await dialog.getByText('大模型配置 · 系统').waitFor()
    assert.equal(resets, 1, 'restore must call DELETE once')
    await page.screenshot({ path: path.join(screenshotDir, '10-config-restored.png'), animations: 'disabled' })

    assert.deepEqual(errors, [], `console/page errors: ${errors.join(' | ')}`)
    console.log('llm config account test passed')
  } finally {
    await browser.close()
  }
})().catch(error => {
  console.error(error)
  process.exit(1)
})
