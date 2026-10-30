// 账号级模型配置：空态、来源标记与“清空配置”浏览器回归。
// Run: BASE_URL=http://127.0.0.1:5174 CHROME_PATH=... node tests/llm-config-account.cjs
const { chromium } = require('playwright')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const screenshotDir = process.env.SCREENSHOT_DIR || path.resolve(__dirname, '../../.screenshots/jwt-auth')
let resets = 0

const unset = () => ({ base_url: '', model: '', api_key_masked: '', api_key_set: false })

function snapshot(sources) {
  return {
    llm: sources.llm === 'account'
      ? { base_url: 'https://account.example/v1', model: 'account-model', api_key_masked: '已设置', api_key_set: true }
      : unset(),
    embedding: { ...unset(), dimensions: null },
    rerank: unset(),
    decision: { ...unset(), mode: 'off' },
    updated_at: '2026-10-30T00:00:00Z',
    index_dimensions: null,
    sources,
  }
}

const allUnset = { llm: 'unset', embedding: 'unset', rerank: 'unset', decision: 'unset' }

;(async () => {
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
          json: snapshot({ llm: 'account', embedding: 'unset', rerank: 'unset', decision: 'unset' }),
        })
      }
      if (pathname === '/api/v1/llm-config' && request.method() === 'DELETE') {
        resets += 1
        return route.fulfill({ json: { ...snapshot(allUnset), reindex_recommended: false } })
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
    await dialog.getByText('当前账号', { exact: false }).first().waitFor()
    await dialog.getByText('Abc1').waitFor()
    await dialog.getByText('大模型配置 · 账号').waitFor()
    await dialog.getByText('Embedding 配置 · 未配置').waitFor()
    await dialog.getByText('配置只保存在当前账号；未填写的分项保持为空').waitFor()
    assert.equal(await dialog.locator('.arc-card').count(), 4, 'must still show 4 config cards')
    assert.equal(
      await dialog.evaluate(element => /sk-[A-Za-z0-9]/.test(element.textContent || '')),
      false,
      'the account dialog must never render a key fragment',
    )
    await page.screenshot({ path: path.join(screenshotDir, '09-config-account.png'), animations: 'disabled' })

    await dialog.getByRole('button', { name: '大模型' }).click()
    const editor = page.getByRole('dialog', { name: '大模型配置' })
    const keyInput = editor.getByLabel('API Key')
    assert.equal(await keyInput.inputValue(), '', 'password input must stay blank')
    assert.equal(await keyInput.getAttribute('placeholder'), '已设置，留空保持不变')
    await page.screenshot({ path: path.join(screenshotDir, '11-config-llm-editor.png'), animations: 'disabled' })
    await editor.getByRole('button', { name: '完成' }).click()
    await editor.waitFor({ state: 'hidden' })

    await dialog.getByRole('button', { name: '清空配置' }).click()
    await dialog.getByText('已清空账号配置', { exact: true }).waitFor()
    await dialog.getByText('大模型配置 · 未配置').waitFor()
    assert.equal(resets, 1, 'clear must call DELETE once')
    await page.screenshot({ path: path.join(screenshotDir, '10-config-cleared.png'), animations: 'disabled' })

    assert.deepEqual(errors, [], `console/page errors: ${errors.join(' | ')}`)
    console.log('llm config account test passed')
  } finally {
    await browser.close()
  }
})().catch(error => {
  console.error(error)
  process.exit(1)
})
