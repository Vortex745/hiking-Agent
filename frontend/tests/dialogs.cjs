// Run against Vite with BASE_URL and optional CHROME_PATH / SCREENSHOT_DIR.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const path = require('node:path');

const configSection = { base_url: 'https://example.com/v1', model: 'fixture-model', api_key_set: true };
const servers = Array.from({ length: 14 }, (_, index) => ({
  id: `server-${index}`, name: `测试服务 ${index}`, builtin: true, transport: 'stdio', status: 'configured', tools: ['fixture'],
}));
const memories = Array.from({ length: 20 }, (_, index) => ({
  memory_id: `memory-${index}`, category: '偏好', content: `长期记忆 ${index}`, updated_at: '2026-10-27T00:00:00Z', expires_at: null,
}));

(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH });
  try {
    for (const width of [390, 768, 1024, 1440]) {
      const colorScheme = process.env.COLOR_SCHEME || 'light';
      const page = await browser.newPage({ viewport: { width, height: width === 390 ? 812 : 900 }, reducedMotion: width === 390 ? 'reduce' : 'no-preference', colorScheme });
      const errors = [];
      page.on('pageerror', error => errors.push(String(error)));
      page.on('console', message => {
        if (/DialogContent|DialogTitle|DialogDescription|Function components cannot be given refs/.test(message.text())) errors.push(message.text());
      });
      await page.route('**/*', route => {
        // JWT 鉴权门 fixture：浏览器测试默认已登录
        if (new URL(route.request().url()).pathname === '/api/auth/me') return route.fulfill({ json: { user_id: 'test-user', login_id: 'tester', created_at: '2026-01-01T00:00:00Z' } });
        const pathname = new URL(route.request().url()).pathname;
        if (pathname === '/api/v1/llm-config') return route.fulfill({ json: { llm: configSection, embedding: { ...configSection, dimensions: 1536 }, rerank: configSection, index_dimensions: 1536 } });
        if (pathname === '/api/v2/mcp/servers') return route.fulfill({ json: { servers } });
        if (pathname === '/api/v2/memory/settings') return route.fulfill({ json: { enabled: true } });
        if (pathname === '/api/v2/memories') return route.fulfill({ json: { items: memories, total: memories.length } });
        return pathname.startsWith('/api/') ? route.fulfill({ json: { enabled: false } }) : route.continue();
      });
      await page.addInitScript(() => {
        navigator.geolocation.getCurrentPosition = (_success, failure) => failure?.({ code: 1, message: 'fixture' });
        const messages = [{ role: 'user', content: '图片测试' }, { role: 'assistant', content: '已收到' }];
        const session = { id: 'dialog-fixture', chatId: 'chat-dialog-fixture', title: '图片测试', date: '2026-10-27', messages };
        localStorage.setItem('ai-hiking-agent-sessions', JSON.stringify([session]));
        localStorage.setItem('ai-hiking-agent-active-session', session.id);
        localStorage.setItem('ai-hiking-agent-chat-id', session.chatId);
        localStorage.setItem('ai-hiking-agent-chat', JSON.stringify(messages));
      });
      await page.goto(process.env.BASE_URL || 'http://127.0.0.1:5173');
      assert.equal(await page.locator('html').getAttribute('data-theme'), colorScheme);
      const otherScheme = colorScheme === 'dark' ? 'light' : 'dark';
      await page.emulateMedia({ colorScheme: otherScheme });
      await page.waitForFunction(theme => document.documentElement.dataset.theme === theme, otherScheme);
      await page.emulateMedia({ colorScheme });
      await page.waitForFunction(theme => document.documentElement.dataset.theme === theme, colorScheme);
      const settings = page.getByRole('button', { name: '设置' });
      const openSettings = async name => {
        await settings.click();
        await page.getByRole('menuitem', { name }).click();
      };
      const checkBounds = async dialog => {
        const box = await dialog.boundingBox();
        assert(box && box.x >= 0 && box.y >= 0 && box.x + box.width <= width && box.y + box.height <= page.viewportSize().height, `dialog exceeds ${width}px viewport`);
        assert(Math.abs(box.x + box.width / 2 - width / 2) < 2, `dialog is not centered at ${width}px`);
        assert.equal(await dialog.evaluate(el => el.scrollWidth <= el.clientWidth), true, `dialog overflows horizontally at ${width}px`);
      };
      const checkClosed = async () => {
        await settings.waitFor({ state: 'visible' });
        await page.waitForFunction(() => getComputedStyle(document.body).pointerEvents === 'auto');
        assert.equal(await settings.evaluate(el => document.activeElement === el), true, 'focus must return to settings');
      };

      await openSettings('MCP 接入');
      let dialog = page.getByRole('dialog', { name: 'MCP 接入' });
      await dialog.getByText('测试服务 13').waitFor({ state: 'attached' });
      await checkBounds(dialog);
      const scroll = await dialog.evaluate(el => [...el.querySelectorAll('*')].find(node => node.scrollHeight > node.clientHeight + 5 && getComputedStyle(node).overflowY === 'auto')?.scrollHeight);
      assert(scroll, 'long MCP list must scroll inside the dialog');
      if (process.env.SCREENSHOT_DIR && (width === 390 || width === 1440)) await page.screenshot({ path: path.join(process.env.SCREENSHOT_DIR, `mcp-dialog-${width}-${colorScheme}.png`) });
      await dialog.getByRole('button', { name: '添加服务' }).click();
      dialog = page.getByRole('dialog', { name: '添加 MCP 服务' });
      await dialog.getByText('stdio', { exact: true }).click();
      await dialog.getByLabel('命令', { exact: true }).waitFor();
      await checkBounds(dialog);
      if (process.env.SCREENSHOT_DIR && (width === 390 || width === 1440)) await page.screenshot({ path: path.join(process.env.SCREENSHOT_DIR, `mcp-editor-${width}-${colorScheme}.png`) });
      await page.keyboard.press('Escape');
      await checkClosed();

      await openSettings('长期记忆');
      dialog = page.getByRole('dialog', { name: '长期记忆' });
      await dialog.getByText('长期记忆 19').waitFor({ state: 'attached' });
      await checkBounds(dialog);
      await dialog.getByRole('button', { name: '关闭' }).click();
      await checkClosed();

      await openSettings('模型配置');
      const parent = page.getByRole('dialog', { name: '模型配置' });
      const modelButton = parent.getByRole('button', { name: '大模型', exact: true });
      await modelButton.waitFor();
      await checkBounds(parent);
      if (process.env.SCREENSHOT_DIR && (width === 390 || width === 1440)) await page.screenshot({ path: path.join(process.env.SCREENSHOT_DIR, `model-dialog-${width}-${colorScheme}.png`) });
      await modelButton.click();
      const editor = page.getByRole('dialog', { name: '大模型配置' });
      await editor.getByLabel('Base URL').waitFor();
      await checkBounds(editor);
      assert.equal(await editor.getByLabel('Base URL').evaluate(el => document.activeElement === el), true, 'editor must focus its first field');
      for (let index = 0; index < 8; index++) {
        await page.keyboard.press('Tab');
        assert.equal(await editor.evaluate(el => el.contains(document.activeElement)), true, 'focus must stay inside nested dialog');
      }
      const overlayAboveParent = await page.evaluate(() => {
        const parent = [...document.querySelectorAll('[role="dialog"]')].find(el => el.textContent.includes('模型配置'));
        const box = parent.getBoundingClientRect();
        return !parent.contains(document.elementFromPoint(box.left + 8, box.top + 8));
      });
      assert.equal(overlayAboveParent, true, 'nested dialog overlay must cover its parent');
      await page.keyboard.press('Escape');
      await editor.waitFor({ state: 'hidden' });
      await modelButton.waitFor();
      assert.equal(await modelButton.evaluate(el => document.activeElement === el), true, 'focus must return to model card');
      await page.mouse.click(4, 4);
      await checkClosed();

      assert.deepEqual(errors, []);
      await page.close();
    }
    console.log('PASS: Arc dialogs at 390, 768, 1024 and 1440px; scrolling, nested overlays, focus and Escape.');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
