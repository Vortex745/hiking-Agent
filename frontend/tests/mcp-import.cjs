// Run against the Vite server with BASE_URL and optional CHROME_PATH.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const path = require('node:path');

(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH });
  try {
    for (const viewport of [{ width: 1440, height: 900 }, { width: 1024, height: 768 }, { width: 768, height: 1024 }, { width: 390, height: 844 }]) {
      for (const colorScheme of ['light', 'dark']) {
        const page = await browser.newPage({ viewport, colorScheme, reducedMotion: 'reduce' });
        const errors = [];
        const requests = [];
        const servers = [{ id: 'builtin', name: '高德地图', builtin: true, transport: 'stdio', status: 'configured', tools: ['route'] }];
        page.on('pageerror', error => errors.push(String(error)));
        page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
        page.setDefaultTimeout(10000);
        await page.route('**/*', route => {
          const request = route.request();
          const pathname = new URL(request.url()).pathname;
          if (pathname === '/api/v2/mcp/servers/test') {
            requests.push({ action: 'test', body: request.postDataJSON() });
            return route.fulfill({ json: { test_id: 'verified', tools: ['echo'] } });
          }
          if (pathname === '/api/v2/mcp/servers' && request.method() === 'GET') return route.fulfill({ json: { servers } });
          if (pathname === '/api/v2/mcp/servers' && request.method() === 'POST') {
            const body = request.postDataJSON();
            requests.push({ action: 'save', body });
            const server = { id: `custom_${servers.length}`, ...body, builtin: false, status: body.test_id ? 'success' : 'untested', tools: body.test_id ? ['echo'] : [] };
            servers.push(server);
            return route.fulfill({ json: server });
          }
          if (pathname.startsWith('/api/v2/mcp/servers/') && request.method() === 'PUT') {
            const body = request.postDataJSON();
            requests.push({ action: 'update', body });
            const server = servers.find(item => pathname.endsWith(item.id));
            Object.assign(server, body);
            return route.fulfill({ json: server });
          }
          return pathname.startsWith('/api/') ? route.fulfill({ json: { enabled: false } }) : route.continue();
        });
        await page.goto(process.env.BASE_URL || 'http://127.0.0.1:5176');
        const settings = page.getByRole('button', { name: '设置' });
        await settings.click();
        await page.getByRole('menuitem', { name: 'MCP 接入' }).click();
        await page.getByRole('dialog', { name: 'MCP 接入' }).getByRole('button', { name: '添加服务' }).click();
        const editor = () => page.getByRole('dialog', { name: '添加 MCP 服务' });
        const importer = () => page.getByRole('dialog', { name: '粘贴 MCP 配置' });
        await editor().getByLabel('名称').fill('已有名称');
        await editor().getByRole('button', { name: '粘贴配置' }).click();
        await importer().getByRole('button', { name: '返回服务配置' }).click();
        assert.equal(await editor().getByLabel('名称').inputValue(), '已有名称');
        await editor().getByRole('button', { name: '粘贴配置' }).click();
        const input = importer().getByLabel('MCP 配置 JSON');
        assert.equal(await input.evaluate(element => document.activeElement === element), true);
        if (process.env.MCP_SCREENSHOT_DIR) await page.screenshot({ path: path.join(process.env.MCP_SCREENSHOT_DIR, `mcp-import-${viewport.width}-${colorScheme}.png`) });

        const invalid = [
          ['{', 'JSON 格式无效'],
          [JSON.stringify({ mcpServers: {} }), '没有 MCP 服务'],
          [JSON.stringify({ mcpServers: { bad: { type: 'sse', url: 'https://example.com' } } }), '仅支持 HTTP 和 stdio'],
          [JSON.stringify({ mcpServers: { bad: { command: 'npx', cwd: '/tmp' } } }), '暂不支持配置项：cwd'],
          [JSON.stringify({ mcpServers: { bad: { command: 'npx', args: [123] } } }), 'args 必须是字符串数组'],
          [JSON.stringify({ mcpServers: { bad: { command: 'npx', args: null } } }), 'args 必须是字符串数组'],
          [JSON.stringify({ mcpServers: { bad: { command: 'npx', env: { TOKEN: 1 } } } }), 'env 必须是字符串键值对象'],
          [JSON.stringify({ servers: { bad: { type: 'http', url: 'https://example.com', headers: { Authorization: '${input:token}' } } } }), '暂不支持变量引用'],
          [JSON.stringify({ inputs: [], servers: { bad: { command: 'npx' } } }), '暂不支持配置项：inputs'],
          [JSON.stringify({ mcpServers: { bad: { type: 'http', url: 'https://example.com', command: 'npx' } } }), '连接方式与配置字段不匹配'],
          [JSON.stringify({ mcpServers: { bad: { command: 'npx', ['unsupported'.repeat(15)]: true } } }), '暂不支持配置项'],
        ];
        for (const [text, expected] of invalid) {
          await input.fill(text);
          await importer().getByRole('button', { name: '导入配置' }).click();
          await importer().getByRole('alert').getByText(expected, { exact: false }).first().waitFor();
          assert.equal(await input.getAttribute('aria-invalid'), 'true');
          const size = await importer().evaluate(element => ({ width: element.scrollWidth, client: element.clientWidth }));
          assert(size.width <= size.client, 'validation message must not overflow the dialog');
        }
        assert.equal(requests.length, 0, 'parsing invalid or pasted JSON must not call the MCP API');

        const stdio = { mcpServers: { 'example-server': { command: 'npx', args: ['-y', 'mcp-server-example'], env: { TOKEN: 'stdio-secret' } } } };
        await input.fill(JSON.stringify(stdio));
        await importer().getByRole('button', { name: '导入配置' }).click();
        assert.equal(await editor().getByLabel('名称').inputValue(), 'example-server');
        assert.equal(await editor().getByLabel('命令', { exact: true }).inputValue(), 'npx');
        assert.deepEqual(JSON.parse(await editor().getByLabel('参数（JSON 数组）').inputValue()), ['-y', 'mcp-server-example']);
        assert.deepEqual(JSON.parse(await editor().getByLabel('环境变量（JSON 对象）').inputValue()), { TOKEN: 'stdio-secret' });
        assert.equal(await editor().getByLabel('允许在本机启动此命令进行测试').isChecked(), false);
        assert.equal(requests.length, 0, 'importing must not test or save');
        await editor().getByRole('button', { name: '测试连接' }).click();
        await editor().getByRole('alert').getByText('请先确认允许在本机启动命令').waitFor();
        assert.equal(requests.length, 0, 'stdio cannot start before confirmation');
        await editor().getByLabel('允许在本机启动此命令进行测试').check();
        await editor().getByRole('button', { name: '测试连接' }).click();
        await editor().getByRole('status').getByText('测试成功', { exact: false }).waitFor();
        assert.equal(requests[0].body.stdio_confirmed, true);
        assert.deepEqual(requests[0].body.env, { TOKEN: 'stdio-secret' });
        await editor().getByRole('button', { name: '保存' }).click();
        let list = page.getByRole('dialog', { name: 'MCP 接入' });
        await list.getByRole('status').getByText('服务已保存，可在列表中启用。').waitFor();
        assert.equal(requests[1].body.enabled, false, 'saved services start disabled');
        assert.equal(requests[1].body.test_id, 'verified');
        await list.getByRole('switch', { name: '启用 example-server' }).click();
        await page.waitForFunction(() => document.querySelector('[role="switch"][aria-label="启用 example-server"]')?.getAttribute('aria-checked') === 'true');
        assert.equal(requests[2].body.enabled, true);

        await list.getByRole('button', { name: '添加服务' }).click();
        await editor().getByRole('button', { name: '粘贴配置' }).click();
        const multiple = { servers: { local: { type: 'stdio', command: 'npx', args: ['example'] }, web: { type: 'http', url: 'https://example.com/mcp', headers: { 'X-Example': 'test-value' } } } };
        await importer().getByLabel('MCP 配置 JSON').fill(JSON.stringify(multiple));
        await importer().getByRole('button', { name: '导入配置' }).click();
        assert.equal(await importer().getByRole('button', { name: '导入所选服务' }).isDisabled(), true);
        assert.equal(await importer().getByRole('radio').count(), 2);
        await importer().getByLabel('MCP 配置 JSON').focus();
        await page.keyboard.press('Tab');
        const firstRadio = importer().getByRole('radio', { name: 'local' });
        assert.equal(await firstRadio.evaluate(element => document.activeElement === element), true);
        assert.notEqual(await firstRadio.evaluate(element => getComputedStyle(element.closest('label')).outlineStyle), 'none');
        await page.keyboard.press('ArrowDown');
        assert.equal(await importer().getByRole('radio', { name: 'web' }).isChecked(), true);
        await importer().getByText('local', { exact: true }).click();
        await importer().getByText('web', { exact: true }).click();
        assert.equal(await importer().getByRole('radio', { name: 'web' }).isChecked(), true);
        if (process.env.MCP_SCREENSHOT_DIR) await page.screenshot({ path: path.join(process.env.MCP_SCREENSHOT_DIR, `mcp-choices-${viewport.width}-${colorScheme}.png`) });
        await importer().getByRole('button', { name: '导入所选服务' }).click();
        assert.equal(await editor().getByLabel('名称').inputValue(), 'web');
        assert.equal(await editor().getByRole('radio', { name: 'HTTP' }).isChecked(), true);
        assert.equal(await editor().getByLabel('服务 URL').inputValue(), 'https://example.com/mcp');
        assert.deepEqual(JSON.parse(await editor().getByLabel('请求头（JSON 对象）').inputValue()), { 'X-Example': 'test-value' });
        assert.equal(requests.length, 3, 'choosing a server must not start a connection');
        await editor().getByRole('button', { name: '测试连接' }).click();
        await editor().getByRole('status').getByText('测试成功', { exact: false }).waitFor();
        assert.deepEqual(requests[3].body.headers, { 'X-Example': 'test-value' });
        await editor().getByRole('button', { name: '保存' }).click();
        list = page.getByRole('dialog', { name: 'MCP 接入' });
        await list.getByText('web', { exact: true }).waitFor();
        assert.equal(requests[4].body.enabled, false);
        await list.getByRole('button', { name: '添加服务' }).click();
        await editor().getByRole('button', { name: '粘贴配置' }).click();
        await importer().getByLabel('MCP 配置 JSON').fill(JSON.stringify({ name: 'single', transport: 'http', url: 'https://example.com/mcp' }));
        await importer().getByRole('button', { name: '导入配置' }).click();
        assert.equal(await editor().getByLabel('名称').inputValue(), 'single');
        const dimensions = await editor().evaluate(element => ({ width: element.scrollWidth, client: element.clientWidth }));
        assert(dimensions.width <= dimensions.client, 'dialog must not overflow');
        await page.keyboard.press('Escape');
        await editor().waitFor({ state: 'hidden' });
        assert.equal(await page.evaluate(() => getComputedStyle(document.body).pointerEvents), 'auto');
        assert.deepEqual(errors, []);
        await page.close();
      }
    }
    console.log('PASS: MCP JSON import on desktop/mobile in both themes.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
