// Run against the Vite server with BASE_URL and optional CHROME_PATH.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');

(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, reducedMotion: 'reduce' });
    page.setDefaultTimeout(10000);
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    await page.addInitScript(() => {
      window.__opened = [];
      window.open = url => {
        window.__opened.push(String(url));
        return null;
      };
    });
    let polls = 0;
    let savedDraft;
    await page.route('**/*', route => {
      // JWT 鉴权门 fixture：浏览器测试默认已登录
      if (new URL(route.request().url()).pathname === '/api/auth/me') return route.fulfill({ json: { user_id: 'test-user', login_id: 'tester', created_at: '2026-01-01T00:00:00Z' } });
      const pathname = new URL(route.request().url()).pathname;
      if (pathname === '/api/v2/mcp/servers') {
        if (route.request().method() === 'POST') {
          savedDraft = route.request().postDataJSON();
          return route.fulfill({ json: { id: 'custom_oauth', name: savedDraft.name, builtin: false, transport: 'http', status: 'success', tools: ['echo'], enabled: false } });
        }
        return route.fulfill({ json: { servers: [
          { id: 'amap_weather', name: '高德天气', builtin: true, transport: 'stdio', status: 'configured', tools: [] },
          { id: 'amap_map', name: '高德地图', builtin: true, transport: 'stdio', status: 'configured', tools: [] },
        ] } });
      }
      if (pathname === '/api/v2/mcp/servers/test') {
        return route.fulfill({ json: { status: 'auth_required', auth_url: 'http://127.0.0.1:8201/authorize?state=test-state', flow_id: 'flow-1' } });
      }
      if (pathname === '/api/v2/mcp/oauth/status/flow-1') {
        polls += 1;
        return route.fulfill({ json: polls < 2 ? { status: 'auth_required' } : { status: 'success', test_id: 'test-1', tools: ['echo'] } });
      }
      return pathname.startsWith('/api/') ? route.fulfill({ json: {} }) : route.continue();
    });

    await page.goto(process.env.BASE_URL || 'http://127.0.0.1:5173');
    await page.getByRole('button', { name: '设置' }).click();
    await page.getByRole('menuitem', { name: 'MCP 接入' }).click();
    await page.getByRole('dialog', { name: 'MCP 接入' }).getByRole('button', { name: '添加服务' }).click();

    const dialog = page.getByRole('dialog', { name: '添加 MCP 服务' });
    await dialog.getByLabel('名称', { exact: true }).fill('OAuth 服务');
    await dialog.getByLabel('服务 URL', { exact: true }).fill('http://127.0.0.1:8201/mcp');
    await dialog.getByRole('button', { name: '测试连接' }).click();

    await dialog.getByText('正在等待 OAuth 授权').waitFor();
    assert.equal(await dialog.getByRole('button', { name: '测试连接' }).isDisabled(), true, 'test button must pause while awaiting authorization');
    assert.equal(await dialog.getByRole('button', { name: '保存' }).isDisabled(), true, 'save button must pause while awaiting authorization');
    assert.deepEqual(await page.evaluate(() => window.__opened), ['http://127.0.0.1:8201/authorize?state=test-state']);

    await dialog.getByRole('button', { name: '打开授权页' }).click();
    assert.deepEqual(await page.evaluate(() => window.__opened), [
      'http://127.0.0.1:8201/authorize?state=test-state',
      'http://127.0.0.1:8201/authorize?state=test-state',
    ]);

    await dialog.getByText('测试成功，发现 1 个工具').waitFor();
    assert.equal(await dialog.getByRole('button', { name: '保存' }).isEnabled(), true);
    assert.equal(await dialog.getByText('正在等待 OAuth 授权').count(), 0, 'authorization notice must clear after success');

    await dialog.getByRole('button', { name: '保存' }).click();
    assert.equal(savedDraft.test_id, 'test-1', 'the OAuth test result must grant the save');
    assert.equal(savedDraft.transport, 'http');
    assert.equal(savedDraft.url, 'http://127.0.0.1:8201/mcp');
    assert.deepEqual(errors, []);
    console.log('PASS: MCP OAuth settings pause, auto-open attempt, polling completion and save grant.');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
