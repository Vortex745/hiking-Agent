// Run against the Vite server with BASE_URL and optional CHROME_PATH.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');

const servers = [
  { id: 'amap_weather', name: '高德天气', builtin: true, transport: 'stdio', status: 'configured', tools: ['weather'] },
  { id: 'amap_map', name: '高德地图', builtin: true, transport: 'stdio', status: 'configured', tools: ['route'] },
  { id: 'custom_test', name: '自定义服务', builtin: false, transport: 'http', url: 'http://127.0.0.1:1/mcp', status: 'success', tools: ['echo'], enabled: true },
];

(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH });
  try {
    for (const viewport of [{ width: 1280, height: 900 }, { width: 375, height: 812 }]) {
      const page = await browser.newPage({ viewport, reducedMotion: 'reduce' });
      const errors = [];
      let available = true;
      let savedDraft;
      page.setDefaultTimeout(10000);
      page.on('pageerror', error => errors.push(String(error)));
      await page.route('**/*', route => {
        const pathname = new URL(route.request().url()).pathname;
        if (pathname === '/api/v2/mcp/servers') {
          return available ? route.fulfill({ json: { servers } }) : route.fulfill({ status: 403, json: { detail: 'MCP 管理仅在本机模式下可用' } });
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
