// Current-account map configuration and SDK lifecycle, using synthetic credentials only.
const { chromium } = require('playwright')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const output = process.env.SCREENSHOT_DIR || path.resolve(__dirname, '../../.screenshots/map-account-isolation')
const base = process.env.BASE_URL || 'http://localhost:5174'
// Only use this mode with an isolated backend seeded with MapA/MapB test accounts.
const realBackend = process.env.MAP_REAL_BACKEND === '1'
const user = name => ({ user_id: `map-${name}`, login_id: name, created_at: '2026-10-30T00:00:00Z' })
const sdkFixture = `
  window.AMap = {
    Geolocation: class { getCurrentPosition(callback) {
      parent.__mapCalls.locations++;
      callback('complete', { position: { lng: 116.4, lat: 39.9 }, accuracy: 25 });
    } },
    PlaceSearch: class { search(keyword, callback) {
      callback('complete', { poiList: { pois: [{ name: keyword, location: { lng: 116.4, lat: 39.9 } }] } });
    } },
    Map: class { constructor(container) {
      this.container = container;
      container.textContent = '地图测试画布';
      container.style.background = '#e5eee6';
      parent.__mapCalls.keys.push(new URL(document.currentScript.src).searchParams.get('key'));
      parent.__mapCalls.codes.push(window._AMapSecurityConfig.securityJsCode);
    }
      on(event, callback) { if (event === 'tilesloaded') setTimeout(callback, 0); }
      add() {} resize() {} destroy() { parent.__mapCalls.destroyed++; }
    },
    Marker: class {}, Circle: class {},
  };
  window.onAMapLoaded();
`

;(async () => {
  fs.mkdirSync(output, { recursive: true })
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH })
  const results = []
  try {
    for (const width of [1280, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 900 }, reducedMotion: 'reduce' })
      page.setDefaultTimeout(12000)
      let account = 'MapA'
      const envs = { MapA: {}, MapB: {} }
      let configError = false
      let mapRequests = 0
      const errors = []
      if (realBackend) {
        for (const login_id of ['MapB', 'MapA']) {
          assert.equal((await page.request.post(`${base}/api/auth/login`, {
            data: { login_id, password: 'password123' },
          })).status(), 200)
          assert.equal((await page.request.delete(`${base}/api/v2/mcp/builtin/credentials`)).status(), 200)
        }
      }
      page.on('pageerror', error => errors.push(error.message))
      await page.addInitScript(() => {
        window.__mapCalls = { locations: 0, keys: [], codes: [], destroyed: 0 }
      })
      const servers = () => ['amap_weather', 'amap_map'].map(id => ({
        id, name: id === 'amap_map' ? '高德地图' : '高德天气', builtin: true,
        transport: 'stdio', status: envs[account].AMAP_API_KEY ? 'configured' : 'unconfigured',
        command: 'npx', args: [], tools: [], env_names: ['AMAP_API_KEY'],
        browser_configured: id === 'amap_map' && Boolean(envs[account].AMAP_JSAPI_KEY),
      }))
      await page.route('**/*', async route => {
        const url = new URL(route.request().url())
        if (url.hostname.endsWith('amap.com')) {
          mapRequests++
          if (url.pathname === '/maps') return route.fulfill({ contentType: 'application/javascript', body: sdkFixture })
          return route.abort()
        }
        const p = url.pathname
        if (p === '/api/auth/me') return realBackend ? route.continue() : route.fulfill({ json: user(account) })
        if (p === '/api/auth/logout') { account = null; return realBackend ? route.continue() : route.fulfill({ json: { ok: true } }) }
        if (p === '/api/auth/login') {
          account = route.request().postDataJSON().login_id
          return realBackend ? route.continue() : route.fulfill({ json: user(account) })
        }
        if (p === '/api/v2/mcp/amap/browser-config') return realBackend && !configError ? route.continue() : route.fulfill({
          status: configError ? 500 : 200,
          json: configError ? { detail: '配置读取失败' } : {
            key: envs[account].AMAP_JSAPI_KEY || '', security_code: envs[account].AMAP_JSAPI_SECURITY_CODE || '',
          },
        })
        if (p === '/api/v2/mcp/servers') return realBackend ? route.continue() : route.fulfill({ json: { servers: servers() } })
        if (p === '/api/v2/mcp/builtin/credentials') {
          if (route.request().method() === 'DELETE') envs[account] = {}
          else Object.assign(envs[account], route.request().postDataJSON().env)
          return realBackend ? route.continue() : route.fulfill({ json: { servers: servers() } })
        }
        return p.startsWith('/api/') ? route.fulfill({ json: { enabled: false } }) : route.continue()
      })
      await page.goto(base)
      const ball = page.getByRole('button', { name: '查看当前位置', exact: true })
      const dialog = page.getByRole('dialog', { name: '当前位置地图', exact: true })
      const closeMap = () => page.getByRole('button', { name: '关闭地图' }).click()
      const openSettings = async () => {
        await page.getByRole('button', { name: '设置', exact: true }).click()
        await page.getByRole('menuitem', { name: 'MCP 接入' }).click()
        await page.getByRole('button', { name: '配置 高德地图 的 Key' }).click()
        return page.getByRole('dialog', { name: '配置 高德地图', exact: true })
      }
      const saveMap = async suffix => {
        const editor = await openSettings()
        await editor.getByLabel('地图 JSAPI Key', { exact: true }).fill(`js-${suffix}`)
        await editor.getByLabel('地图安全密钥', { exact: true }).fill(`code-${suffix}`)
        await editor.getByRole('button', { name: '保存', exact: true }).click()
        await page.getByText('已保存内置服务 Key。', { exact: true }).waitFor()
        await page.getByRole('dialog', { name: 'MCP 接入', exact: true }).getByRole('button', { name: '关闭', exact: true }).click()
      }
      const switchAccount = async name => {
        if (width < 768) await page.getByRole('button', { name: '打开对话历史', exact: true }).click()
        await page.getByRole('button', { name: `账号 ${account}` }).click()
        await page.getByRole('menuitem', { name: '退出登录' }).click()
        await page.locator('input[name="loginId"]').fill(name)
        await page.locator('input[name="password"]').fill('password123')
        await page.getByRole('button', { name: '登录', exact: true }).click()
        await ball.waitFor()
      }
      const assertEmpty = async () => {
        const before = mapRequests
        await ball.click()
        await dialog.getByText('请配置高德地图 API', { exact: true }).waitFor()
        assert.equal(await page.locator('iframe[title="高德地图"]').count(), 0)
        assert.equal(mapRequests, before, 'empty account must not load an AMap script')
      }
      const assertMap = async suffix => {
        await ball.click()
        await page.frameLocator('iframe[title="高德地图"]').getByText('地图测试画布').waitFor()
        await page.waitForFunction(() => !document.querySelector('[role="dialog"] .animate-spin'))
        assert.equal(await page.evaluate(() => window.__mapCalls.keys.at(-1)), `js-${suffix}`)
        assert.equal(await page.evaluate(() => window.__mapCalls.codes.at(-1)), `code-${suffix}`)
        assert.equal(await page.evaluate(() => Boolean(window.AMap)), false, 'SDK must stay in its own frame')
      }

      await assertEmpty()
      assert.equal(await page.evaluate(() => window.__mapCalls.locations), 0)
      await page.screenshot({ path: path.join(output, `unconfigured-${width}.png`) })
      await closeMap()
      // A web-service Key alone must not enable the JSAPI map.
      envs.MapA.AMAP_API_KEY = 'web-service-only'
      if (realBackend) assert.equal((await page.request.put(`${base}/api/v2/mcp/builtin/credentials`, {
        data: { env: { AMAP_API_KEY: 'web-service-only' } },
      })).status(), 200)
      await assertEmpty()
      await closeMap()
      await saveMap('a')
      assert.equal(envs.MapA.AMAP_API_KEY, 'web-service-only', 'saving map must preserve the service Key')
      await assertMap('a')
      await closeMap()
      assert.equal(await page.locator('iframe[title="高德地图"]').count(), 0)
      await page.waitForFunction(() => window.__mapCalls.destroyed === 1)
      assert.equal(await page.evaluate(() => window.__mapCalls.destroyed), 1)
      await switchAccount('MapB')
      await assertEmpty()
      await closeMap()
      await saveMap('b')
      await assertMap('b')
      // Esc also works while focus is inside the map frame.
      await page.frameLocator('iframe[title="高德地图"]').getByText('地图测试画布').click()
      await page.keyboard.press('Escape')
      await ball.waitFor()
      await switchAccount('MapA')
      await assertMap('a')
      await closeMap()
      await saveMap('a-replaced')
      await assertMap('a-replaced')
      // Clear while the map is open: the successful save event must destroy it immediately.
      const editor = await openSettings()
      await editor.getByRole('button', { name: '清除 Key' }).click()
      await page.getByText('已清除内置服务 Key。', { exact: true }).waitFor()
      await page.getByRole('dialog', { name: 'MCP 接入', exact: true }).getByRole('button', { name: '关闭', exact: true }).click()
      await dialog.getByText('请配置高德地图 API', { exact: true }).waitFor()
      assert.equal(await page.locator('iframe[title="高德地图"]').count(), 0)
      assert.equal(envs.MapB.AMAP_JSAPI_KEY, 'js-b')
      assert.deepEqual(await page.evaluate(() => window.__mapCalls.keys), ['js-a', 'js-b', 'js-a', 'js-a-replaced'])
      await closeMap()
      configError = true
      const beforeError = mapRequests
      await ball.click()
      await dialog.getByText('地图暂时不可用', { exact: true }).waitFor()
      assert.equal(mapRequests, beforeError, 'a configuration API failure must not use another Key')
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
      assert.deepEqual(errors, [])
      results.push({ width, passed: true, mapRequests })
      console.log(`PASS: map account checks at ${width}px`)
      await page.close()
    }
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify(results, null, 2))
    console.log('PASS: empty/service-only accounts blocked, A→B→A, config replacement, live clear, frame cleanup/Escape, API failure, desktop/mobile.')
  } finally { await browser.close() }
})().catch(error => { console.error(error); process.exitCode = 1 })
