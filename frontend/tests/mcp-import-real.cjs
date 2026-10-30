// End-to-end smoke test using isolated local MCP fixtures and configuration.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const net = require('node:net');
const path = require('node:path');

const root = path.resolve(__dirname, '../..');
const service = path.join(root, 'ai-service');
const python = path.join(service, '.venv', 'Scripts', 'python.exe');
const fixture = path.join(service, 'test_fixtures', 'mcp_server.py');

async function freePort() {
  const server = net.createServer();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const port = server.address().port;
  await new Promise(resolve => server.close(resolve));
  return port;
}

async function waitFor(url, process) {
  for (let attempt = 0; attempt < 150; attempt++) {
    if (process.exitCode !== null) throw new Error(`Test server exited: ${process.exitCode}`);
    try { await fetch(url); return; } catch { await new Promise(resolve => setTimeout(resolve, 100)); }
  }
  throw new Error(`Test server did not start: ${url}`);
}

(async () => {
  const tempParent = path.join(service, '.tmp-harness-test');
  fs.mkdirSync(tempParent, { recursive: true });
  const temp = fs.mkdtempSync(path.join(tempParent, 'mcp-import-'));
  const apiPort = await freePort();
  const mcpPort = await freePort();
  const apiCode = [
    'import sys, uvicorn',
    'from fastapi import FastAPI',
    'from api.mcp_router import mcp_router',
    'from config import settings',
    'settings.agent_v2_state_dir = sys.argv[1]',
    'settings.agent_v2_sandbox_type = "local_native"',
    'app = FastAPI()',
    'app.include_router(mcp_router, prefix="/api/v2")',
    'uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[2]), log_level="error")',
  ].join('\n');
  const api = spawn(python, ['-u', '-c', apiCode, temp, String(apiPort)], { cwd: service, stdio: 'ignore', windowsHide: true });
  const mcp = spawn(python, ['-u', fixture, '--http-port', String(mcpPort)], { cwd: service, stdio: 'ignore', windowsHide: true });
  const exits = [api, mcp].map(process => new Promise(resolve => process.once('exit', resolve)));
  let browser;
  try {
    const apiUrl = `http://127.0.0.1:${apiPort}`;
    await Promise.all([waitFor(`${apiUrl}/api/v2/mcp/servers`, api), waitFor(`http://127.0.0.1:${mcpPort}/mcp`, mcp)]);
    browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH });
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, reducedMotion: 'reduce' });
    const mutations = [];
    await page.route('**/*', async route => {
      const request = route.request();
      const pathname = new URL(request.url()).pathname;
      if (pathname.startsWith('/api/v2/mcp/servers')) {
        if (request.method() !== 'GET') mutations.push(pathname);
        return route.fulfill({ response: await route.fetch({ url: `${apiUrl}${pathname}` }) });
      }
      return pathname.startsWith('/api/') ? route.fulfill({ json: { enabled: false } }) : route.continue();
    });
    await page.goto(process.env.BASE_URL || 'http://127.0.0.1:5176');
    await page.getByRole('button', { name: '设置' }).click();
    await page.getByRole('menuitem', { name: 'MCP 接入' }).click();
    const list = () => page.getByRole('dialog', { name: 'MCP 接入' });
    const editor = () => page.getByRole('dialog', { name: '添加 MCP 服务' });
    const importer = () => page.getByRole('dialog', { name: '粘贴 MCP 配置' });
    const cases = [
      { name: 'fixture-stdio', config: { command: python, args: [fixture], env: { MCP_TEST_EFFECT_FILE: path.join(temp, 'effects') } }, stdio: true },
      { name: 'fixture-http', config: { type: 'http', url: `http://127.0.0.1:${mcpPort}/mcp`, headers: { 'X-Test': 'example' } }, stdio: false },
    ];
    for (const item of cases) {
      await list().getByRole('button', { name: '添加服务' }).click();
      await editor().getByRole('button', { name: '粘贴配置' }).click();
      await importer().getByLabel('MCP 配置 JSON').fill(JSON.stringify({ mcpServers: { [item.name]: item.config } }));
      const before = mutations.length;
      await importer().getByRole('button', { name: '导入配置' }).click();
      assert.equal(mutations.length, before, 'import must not start a connection');
      if (item.stdio) await editor().getByLabel('允许在本机启动此命令进行测试').check();
      await editor().getByRole('button', { name: '测试连接' }).click();
      await editor().getByRole('status').getByText('测试成功，发现 1 个工具').waitFor({ timeout: 20000 });
      await editor().getByRole('button', { name: '保存' }).click();
      await list().getByText(item.name, { exact: true }).waitFor();
      await list().getByRole('switch', { name: `启用 ${item.name}` }).click();
      await page.waitForFunction(name => document.querySelector(`[role="switch"][aria-label="启用 ${name}"]`)?.getAttribute('aria-checked') === 'true', item.name);
    }
    const saved = await (await fetch(`${apiUrl}/api/v2/mcp/servers`)).json();
    for (const item of cases) {
      const server = saved.servers.find(server => server.name === item.name);
      assert.equal(server.enabled, true);
      assert.equal(server.status, 'success');
      assert.deepEqual(server.tools, ['echo']);
    }
    console.log('PASS: pasted stdio and HTTP configs connected to real local MCP fixtures, saved and enabled.');
  } finally {
    if (browser) await browser.close();
    api.kill();
    mcp.kill();
    await Promise.all(exits);
    const config = path.join(temp, 'mcp_servers.json');
    if (fs.existsSync(config)) fs.unlinkSync(config);
    if (fs.readdirSync(temp).length === 0) fs.rmdirSync(temp);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
