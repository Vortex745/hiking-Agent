// JWT 登录/注册/退出 + 账号会话隔离的浏览器回归。
// Run: BASE_URL=http://127.0.0.1:5174 CHROME_PATH=<chrome> node tests/auth.cjs
const { chromium } = require('playwright')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const SCREENSHOT_DIR = process.env.SCREENSHOT_DIR || path.resolve(__dirname, '../../.screenshots/jwt-auth')
const USERS = {
  alice: { user_id: 'user-alice', login_id: 'Abc1', created_at: '2026-10-30T00:00:00Z' },
  bob: { user_id: 'user-bob', login_id: 'Bob123', created_at: '2026-10-30T00:00:00Z' },
}

function json(route, body, status = 200) {
  return route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

(async () => {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true })
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.CHROME_PATH,
  })
  let authedUser = null
  const errors = []
  const baseUrl = process.env.BASE_URL || 'http://127.0.0.1:5174'

  try {
    const context = await browser.newContext({ viewport: { width: 1280, height: 900 } })
    const page = await context.newPage()
    page.on('pageerror', error => errors.push(String(error)))
    page.on('console', message => {
      if (message.type() === 'error' && !/Failed to load resource/.test(message.text())) {
        errors.push(message.text())
      }
    })

    await page.route('**/*', async route => {
      const url = new URL(route.request().url())
      const pathname = url.pathname
      if (pathname === '/api/auth/me') {
        if (!authedUser) {
          return json(route, { detail: { code: 'unauthenticated', message: '请先登录' } }, 401)
        }
        return json(route, authedUser)
      }
      if (pathname === '/api/auth/login') {
        const body = route.request().postDataJSON()
        if (body.login_id === USERS.alice.login_id && body.password === 'password123') {
          authedUser = USERS.alice
          return json(route, { ...USERS.alice, access_token: 'fixture-token', token_type: 'bearer' })
        }
        return json(route, { detail: { code: 'invalid_credentials', message: '登录 ID 或密码错误' } }, 401)
      }
      if (pathname === '/api/auth/register') {
        const body = route.request().postDataJSON()
        if (body.login_id === USERS.alice.login_id) {
          return json(route, { detail: { code: 'duplicate_login_id', message: '该登录 ID 已被使用（区分大小写）' } }, 409)
        }
        authedUser = { ...USERS.bob, login_id: body.login_id }
        return json(route, { ...authedUser, access_token: 'fixture-token', token_type: 'bearer' }, 201)
      }
      if (pathname === '/api/auth/logout') {
        authedUser = null
        return json(route, { ok: true })
      }
      if (pathname === '/api/v2/memory/settings') {
        return json(route, { subject_id: authedUser?.user_id ?? '', enabled: false })
      }
      if (pathname.startsWith('/api/')) {
        return json(route, { enabled: false })
      }
      return route.continue()
    })

    await page.addInitScript(() => {
      const messages = [{ role: 'user', content: '老君山有什么好玩的' }]
      const session = {
        id: 'legacy-session',
        chatId: 'chat-legacy',
        title: '老君山有什么好玩的',
        date: '2026-10-30',
        messages,
      }
      localStorage.setItem('ai-hiking-agent-sessions', JSON.stringify([session]))
      localStorage.setItem('ai-hiking-agent-active-session', session.id)
      localStorage.setItem('ai-hiking-agent-chat-id', session.chatId)
      localStorage.setItem('ai-hiking-agent-chat', JSON.stringify(messages))
    })

    await page.goto(baseUrl, { waitUntil: 'domcontentloaded' })

    // 1. 未登录显示登录页
    await page.getByRole('heading', { name: 'AI Hiking' }).waitFor()
    await page.getByRole('tab', { name: '登录' }).waitFor()
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '01-login.png'), animations: 'disabled' })

    // 2. 客户端校验：ID 太短
    await page.getByLabel('登录 ID').fill('a')
    await page.getByLabel('密码', { exact: true }).fill('password123')
    await page.getByRole('button', { name: '登录', exact: true }).click()
    await page.getByRole('alert').filter({ hasText: '登录 ID' }).waitFor()
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '02-validation.png'), animations: 'disabled' })

    // 3. 登录成功，侧边栏底部显示账号；旧会话搬到当前账号
    await page.getByLabel('登录 ID').fill('Abc1')
    await page.getByLabel('密码', { exact: true }).fill('password123')
    await page.getByRole('button', { name: '登录', exact: true }).click()
    const accountButton = page.getByRole('button', { name: /账号 Abc1/ })
    await accountButton.waitFor()
    await page.getByText('老君山有什么好玩的').first().waitFor()
    const storedSessions = await page.evaluate(() =>
      localStorage.getItem('u:user-alice:ai-hiking-agent-sessions'),
    )
    assert.ok(storedSessions && storedSessions.includes('老君山'), '旧会话应迁移到账号命名空间')
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '03-logged-in.png'), animations: 'disabled' })

    // 4. 点击账号展开面板，退出登录
    await accountButton.click()
    await page.getByRole('menuitem', { name: '退出登录' }).waitFor()
    await page.getByText('当前登录').waitFor()
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '04-account-menu.png'), animations: 'disabled' })
    await page.getByRole('menuitem', { name: '退出登录' }).click()
    await page.getByRole('heading', { name: 'AI Hiking' }).waitFor()
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '05-logged-out.png'), animations: 'disabled' })

    // 5. 注册第二个账号：旧会话不再出现（按账号隔离）
    await page.getByRole('tab', { name: '注册' }).click()
    await page.getByLabel('登录 ID').fill('Bob123')
    await page.getByLabel('密码', { exact: true }).fill('password123')
    await page.getByLabel('确认密码').fill('password123')
    await page.getByRole('button', { name: '创建账号' }).click()
    await page.getByRole('button', { name: /账号 Bob123/ }).waitFor()
    await page.getByText('暂无历史对话').first().waitFor()
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '06-second-account.png'), animations: 'disabled' })

    // 6. 重复 ID 报错（大小写敏感的唯一性由后端保证）
    await page.getByRole('button', { name: /账号 Bob123/ }).click()
    await page.getByRole('menuitem', { name: '退出登录' }).click()
    await page.getByRole('heading', { name: 'AI Hiking' }).waitFor()
    await page.getByRole('tab', { name: '注册' }).click()
    await page.getByLabel('登录 ID').fill('Abc1')
    await page.getByLabel('密码', { exact: true }).fill('password123')
    await page.getByLabel('确认密码').fill('password123')
    await page.getByRole('button', { name: '创建账号' }).click()
    await page.getByRole('alert').filter({ hasText: '已被使用' }).waitFor()
    await page.screenshot({ path: path.join(SCREENSHOT_DIR, '07-duplicate-id.png'), animations: 'disabled' })

    // 7. 移动端：底部账号块在抽屉里
    const mobile = await browser.newPage({ viewport: { width: 390, height: 812 }, reducedMotion: 'reduce' })
    mobile.on('pageerror', error => errors.push(String(error)))
    authedUser = USERS.alice
    await mobile.route('**/*', async route => {
      const url = new URL(route.request().url())
      if (url.pathname === '/api/auth/me') return json(route, authedUser)
      if (url.pathname === '/api/v2/memory/settings') return json(route, { subject_id: authedUser.user_id, enabled: false })
      if (url.pathname.startsWith('/api/')) return json(route, { enabled: false })
      return route.continue()
    })
    await mobile.addInitScript(() => {
      localStorage.setItem('ai-hiking-storage-claimed', 'user-alice')
      localStorage.setItem('u:user-alice:ai-hiking-agent-sessions', JSON.stringify([
        { id: 's1', chatId: 'chat-1', title: '周末徒步', date: '2026-10-30', messages: [] },
      ]))
      localStorage.setItem('u:user-alice:ai-hiking-agent-active-session', 's1')
    })
    await mobile.goto(baseUrl, { waitUntil: 'domcontentloaded' })
    await mobile.getByRole('button', { name: '打开对话历史' }).click()
    await mobile.getByRole('button', { name: /账号 Abc1/ }).waitFor()
    await mobile.getByText('周末徒步').last().waitFor()
    await mobile.screenshot({ path: path.join(SCREENSHOT_DIR, '08-mobile-account.png'), animations: 'disabled' })

    assert.deepEqual(errors, [], `console/page errors: ${errors.join(' | ')}`)
    console.log('auth browser test passed')
  } finally {
    await browser.close()
  }
})().catch(error => {
  console.error(error)
  process.exit(1)
})
