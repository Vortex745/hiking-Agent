// Run against the Vite server with BASE_URL and optional CHROME_PATH.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const screenshotDir = process.env.SCREENSHOT_DIR || path.resolve(__dirname, '../../.screenshots/mcp-settings');

const servers = [
  { id: 'amap_weather', name: '高德天气', builtin: true, transport: 'stdio', status: 'unconfigured', tools: ['weather'], command: 'npx', args: ['-y', '@amap/amap-weather-mcp'], env_names: ['AMAP_API_KEY'] },
  { id: 'amap_map', name: '高德地图', builtin: true, transport: 'stdio', status: 'unconfigured', tools: ['route'], command: 'npx', args: ['-y', '@amap/amap-map-mcp'], env_names: ['AMAP_API_KEY'] },
  { id: 'custom_test', name: '自定义服务', builtin: false, transport: 'http', url: 'http://127.0.0.1:1/mcp', status: 'success', tools: ['echo'], enabled: true },
];

(async () => {
  fs.mkdirSync(screenshotDir, { recursive: true });
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH });
  try {
    for (const viewport of [{ width: 1280, height: 900 }, { width: 375, height: 812 }]) {
      const page = await browser.newPage({ viewport, reducedMotion: 'reduce' });
      const errors = [];
      let available = true;
      let savedDraft;
      let builtinEnv = {};
      const builtinServers = () => servers.map(server => server.builtin
        ? { ...server, status: builtinEnv.AMAP_API_KEY ? 'configured' : 'unconfigured' }
        : server);
      page.setDefaultTimeout(10000);
      page.on('pageerror', error => errors.push(String(error)));
      await page.route('**/*', route => {
        // JWT 鉴权门 fixture：浏览器测试默认已登录
        if (new URL(route.request().url()).pathname === '/api/auth/me') return route.fulfill({ json: { user_id: 'test-user', login_id: 'tester', created_at: '2026-01-01T00:00:00Z' } });
        const pathname = new URL(route.request().url()).pathname;
        if (pathname === '/api/v2/mcp/builtin/credentials') {
          if (route.request().method() === 'PUT') builtinEnv = route.request().postDataJSON().env;
          if (route.request().method() === 'DELETE') builtinEnv = {};
          return route.fulfill({ json: { servers: builtinServers() } });
        }
        if (pathname === '/api/v2/mcp/servers') {
          return available ? route.fulfill({ json: { servers: builtinServers() } }) : route.fulfill({ status: 403, json: { detail: 'MCP 管理仅在本机模式下可用' } });
        }
        if (pathname === '/api/v2/mcp/servers/test') return route.fulfill({ status: 502, json: { detail: '连接失败' } });
        if (pathname === '/api/v2/mcp/servers/custom_test') {
          savedDraft = route.request().postDataJSON();
          return route.fulfill({ json: { ...servers[2], enabled: savedDraft.enabled } });
        }
        return pathname.startsWith('/api/') ? route.fulfill({ json: { enabled: false } }) : route.continue();
      });
      await page.goto(process.env.BASE_URL || 'http://127.0.0.1:5173');
      const settings = page.getByRole('button', { name: '设置' });
      const open = async () => {
        await settings.click();
        await page.getByRole('menuitem', { name: 'MCP 接入' }).click();
      };
      await open();
      let dialog = page.getByRole('dialog', { name: 'MCP 接入' });
      await dialog.getByText('自定义服务', { exact: true }).waitFor();
      assert.deepEqual(await dialog.locator('.mcp-row-name').allTextContents(), ['高德天气', '高德地图', '自定义服务']);
      await page.screenshot({ path: path.join(screenshotDir, `mcp-builtin-unconfigured-${viewport.width}.png`), animations: 'disabled' });
      await dialog.getByRole('button', { name: '配置 高德天气 的 Key' }).click();
      let builtinDialog = page.getByRole('dialog', { name: '配置 高德天气' });
      await builtinDialog.getByLabel('高德 API Key').fill('ak-deadbeef');
      await page.screenshot({ path: path.join(screenshotDir, `mcp-builtin-key-form-${viewport.width}.png`), animations: 'disabled' });
      await builtinDialog.getByRole('button', { name: '保存' }).click();
      dialog = page.getByRole('dialog', { name: 'MCP 接入' });
      await dialog.getByText('已保存内置服务 Key。').waitFor();
      assert.deepEqual(builtinEnv, { AMAP_API_KEY: 'ak-deadbeef' });
      const metas = await dialog.locator('.mcp-row-meta').allTextContents();
      assert(metas[0].includes('已配置') && metas[1].includes('已配置'), metas.join(' | '));
      await page.screenshot({ path: path.join(screenshotDir, `mcp-builtin-configured-${viewport.width}.png`), animations: 'disabled' });
      await dialog.getByRole('button', { name: '配置 高德地图 的 Key' }).click();
      builtinDialog = page.getByRole('dialog', { name: '配置 高德地图' });
      await builtinDialog.getByRole('button', { name: '清除 Key' }).click();
      dialog = page.getByRole('dialog', { name: 'MCP 接入' });
      await dialog.getByText('已清除内置服务 Key。').waitFor();
      assert.deepEqual(builtinEnv, {});
      const clearedMetas = await dialog.locator('.mcp-row-meta').allTextContents();
      assert(clearedMetas[0].includes('未配置') && clearedMetas[1].includes('未配置'), clearedMetas.join(' | '));
      const dimensions = await dialog.evaluate(element => ({ width: element.scrollWidth, client: element.clientWidth }));
      assert(dimensions.width <= dimensions.client, `drawer overflows at ${viewport.width}px`);
      await dialog.getByRole('button', { name: '编辑 自定义服务' }).click();
      let editor = page.getByRole('dialog', { name: '编辑 MCP 服务' });
      await editor.getByRole('button', { name: '测试连接' }).click();
      await editor.getByRole('alert').waitFor();
      await editor.getByRole('button', { name: '保存' }).click();
      await page.getByRole('dialog', { name: 'MCP 接入' }).getByText('自定义服务', { exact: true }).waitFor();
      assert.equal(savedDraft.enabled, false, 'failed test must save the edited service disabled');
      dialog = page.getByRole('dialog', { name: 'MCP 接入' });
      await dialog.getByRole('button', { name: '添加服务' }).click();
      dialog = page.getByRole('dialog', { name: '添加 MCP 服务' });
      await dialog.getByText('stdio', { exact: true }).click();
      assert.equal(await dialog.getByRole('radio', { name: 'stdio' }).isChecked(), true);
      assert.equal(await dialog.getByLabel('命令', { exact: true }).count(), 1);
      assert.equal(await dialog.getByText('允许在本机启动此命令进行测试').count(), 1);
      await page.keyboard.press('Escape');
      await dialog.waitFor({ state: 'hidden' });
      await settings.waitFor({ state: 'visible' });
      assert.equal(await settings.evaluate(element => document.activeElement === element), true);
      assert.equal(await page.evaluate(() => getComputedStyle(document.body).pointerEvents), 'auto');

      available = false;
      await open();
      dialog = page.getByRole('dialog', { name: 'MCP 接入' });
      await dialog.getByRole('alert').waitFor();
      assert.equal(await dialog.getByRole('button', { name: '添加服务' }).isDisabled(), true);
      available = true;
      await dialog.getByRole('button', { name: '重试' }).click();
      await dialog.getByText('自定义服务', { exact: true }).waitFor();
      assert.equal(await dialog.getByRole('button', { name: '添加服务' }).isEnabled(), true);
      await dialog.getByRole('button', { name: '关闭' }).click();
      assert.equal(await page.evaluate(() => getComputedStyle(document.body).pointerEvents), 'auto');
      await settings.click();
      await page.getByRole('menuitem', { name: 'MCP 接入' }).waitFor();
      assert.deepEqual(errors, []);
      await page.close();
    }
    console.log('PASS: MCP settings at desktop/mobile, service order, stdio form, focus, dismissal and unavailable/retry.');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
