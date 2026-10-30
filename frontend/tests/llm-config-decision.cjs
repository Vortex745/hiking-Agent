// Decision-model config browser regression.
// Run: BASE_URL=http://localhost:5173 CHROME_PATH=... SCREENSHOT_DIR=... node tests/llm-config-decision.cjs
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const llm = { base_url: 'https://llm.example/v1', model: 'llm-model', api_key_masked: 'sk-a...0000', api_key_set: true };
const embedding = { ...llm, model: 'emb-model', dimensions: 1536 };
const rerank = { ...llm, model: 'rerank-model' };
let decision = {
  base_url: 'https://api.siliconflow.cn/v1',
  model: 'Kev-4b',
  api_key_masked: '',
  api_key_set: false,
  mode: 'off',
};
let putBody = null;

function snapshot() {
  return {
    llm,
    embedding,
    rerank,
    decision,
    updated_at: '2026-10-29T00:00:00Z',
    index_dimensions: 1536,
  };
}

(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH });
  const screenshotDir = process.env.SCREENSHOT_DIR;
  if (screenshotDir) fs.mkdirSync(screenshotDir, { recursive: true });
  try {
    for (const width of [1440, 390]) {
      decision = {
        base_url: 'https://api.siliconflow.cn/v1',
        model: 'Kev-4b',
        api_key_masked: '',
        api_key_set: false,
        mode: 'off',
      };
      putBody = null;
      const page = await browser.newPage({
        viewport: { width, height: width === 390 ? 844 : 1000 },
        reducedMotion: 'reduce',
        colorScheme: 'light',
      });
      const errors = [];
      page.on('pageerror', error => errors.push(String(error)));
      page.on('console', message => {
        if (message.type() === 'error') errors.push(message.text());
      });

      await page.route('**/*', route => {
        const request = route.request();
        const pathname = new URL(request.url()).pathname;
        if (pathname === '/api/v1/llm-config' && request.method() === 'GET') {
          return route.fulfill({ json: snapshot() });
        }
        if (pathname === '/api/v1/llm-config' && request.method() === 'PUT') {
          putBody = JSON.parse(request.postData() || '{}');
          const next = putBody.decision || {};
          decision = {
            base_url: next.base_url || decision.base_url,
            model: next.model || decision.model,
            api_key_masked: next.api_key ? 'sk-d...7890' : decision.api_key_masked,
            api_key_set: Boolean(next.api_key) || decision.api_key_set,
            mode: next.mode || decision.mode,
          };
          return route.fulfill({ json: { ...snapshot(), reindex_recommended: false } });
        }
        if (pathname === '/api/v1/llm-config/test') {
          const body = JSON.parse(request.postData() || '{}');
          if (body.section === 'decision') {
            return route.fulfill({ json: { latency_ms: 21, model_count: 1, models: ['ok'] } });
          }
          return route.fulfill({ json: { latency_ms: 30, model_count: 2, models: ['m-a', 'm-b'] } });
        }
        if (pathname === '/api/v1/models/fetch') {
          return route.fulfill({
            json: { models: ['Kev-4B', 'SemIf', 'jev-latest', 'jev-1.14.0', 'mimo-v2.6-pro'] },
          });
        }
        if (pathname === '/api/v2/mcp/servers') return route.fulfill({ json: { servers: [] } });
        if (pathname === '/api/v2/memory/settings') return route.fulfill({ json: { enabled: false } });
        return pathname.startsWith('/api/') ? route.fulfill({ json: { enabled: false } }) : route.continue();
      });

      await page.addInitScript(() => {
        navigator.geolocation.getCurrentPosition = (_success, failure) => failure?.({ code: 1, message: 'fixture' });
        const messages = [{ role: 'user', content: '配置测试' }, { role: 'assistant', content: '已收到' }];
        const session = { id: 'llm-config-fixture', chatId: 'chat-llm-config', title: '配置测试', date: '2026-10-29', messages };
        localStorage.setItem('ai-hiking-agent-sessions', JSON.stringify([session]));
        localStorage.setItem('ai-hiking-agent-active-session', session.id);
        localStorage.setItem('ai-hiking-agent-chat-id', session.chatId);
        localStorage.setItem('ai-hiking-agent-chat', JSON.stringify(messages));
      });

      await page.goto(process.env.BASE_URL || 'http://localhost:5173');
      await page.getByRole('button', { name: '设置' }).click();
      await page.getByRole('menuitem', { name: '模型配置' }).click();

      const configDialog = page.getByRole('dialog', { name: '模型配置' });
      await configDialog.waitFor();
      await configDialog.getByRole('button', { name: '决策模型', exact: true }).waitFor();
      assert.equal(await configDialog.locator('.arc-card').count(), 4, 'model config must show 4 cards');

      if (screenshotDir) {
        await configDialog.screenshot({ path: path.join(screenshotDir, `llm-config-decision-${width}.png`) });
      }

      await configDialog.getByRole('button', { name: '决策模型', exact: true }).click();
      const editor = page.getByRole('dialog', { name: '决策模型配置' });
      await editor.waitFor();
      assert.equal(await editor.locator('#decision-base-url').inputValue(), 'https://api.siliconflow.cn/v1');
      assert.equal(await editor.locator('#decision-model').inputValue(), 'Kev-4b');
      assert.equal(await editor.locator('#decision-preset').inputValue(), 'siliconflow');
      await editor.locator('#decision-preset').selectOption('typesafe');
      assert.equal(await editor.locator('#decision-base-url').inputValue(), 'https://api.typesafe.ai/v1');
      assert.equal(await editor.locator('#decision-model').inputValue(), 'jev-latest');
      await editor.locator('#decision-preset').selectOption('siliconflow');
      assert.equal(await editor.locator('#decision-base-url').inputValue(), 'https://api.siliconflow.cn/v1');
      assert.equal(await editor.locator('#decision-model').inputValue(), 'Kev-4b');
      await editor.locator('#decision-api-key').fill('sk-decision-test-123456');
      await editor.getByRole('button', { name: '获取模型' }).click();
      await editor.getByRole('button', { name: 'SemIf', exact: true }).waitFor();
      await editor.getByRole('button', { name: 'SemIf', exact: true }).click();
      assert.equal(await editor.locator('#decision-model').inputValue(), 'SemIf');
      await editor.locator('#decision-mode').selectOption('enforce');
      if (screenshotDir) {
        await editor.screenshot({ path: path.join(screenshotDir, `llm-config-decision-editor-${width}.png`) });
      }
      await editor.getByRole('button', { name: '完成' }).click();

      await configDialog.getByRole('button', { name: '保存' }).click();
      await configDialog.getByText('已保存并生效', { exact: true }).waitFor();
      assert.ok(putBody, 'PUT must be sent');
      assert.equal(putBody.decision.base_url, 'https://api.siliconflow.cn/v1');
      assert.equal(putBody.decision.model, 'SemIf');
      assert.equal(putBody.decision.api_key, 'sk-decision-test-123456');
      assert.equal(putBody.decision.mode, 'enforce');
      await configDialog.getByText('enforce 生效', { exact: true }).waitFor();
      if (screenshotDir) {
        await configDialog.screenshot({ path: path.join(screenshotDir, `llm-config-decision-saved-${width}.png`) });
      }

      await configDialog.getByRole('button', { name: '测速' }).click();
      await configDialog.getByText('决策结果：ok', { exact: true }).waitFor();
      if (screenshotDir) {
        await configDialog.screenshot({ path: path.join(screenshotDir, `llm-config-decision-tested-${width}.png`) });
      }

      assert.deepEqual(errors, [], `console errors at ${width}px: ${errors.join(' | ')}`);
      await page.close();
    }
    console.log('llm-config-decision: PASS');
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exit(1);
});
