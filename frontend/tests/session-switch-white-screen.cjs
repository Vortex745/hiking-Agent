// Regression probe for the session-switch white screen.
// Run with npm run test:browser after npx playwright install chromium.
// CHROME_PATH optionally selects an installed browser instead of Chromium.
// BASE_URL defaults to the project's Vite port.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const errors = [];

(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.CHROME_PATH,
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    page.setDefaultTimeout(5000);
    page.on('pageerror', error => errors.push(error.stack || String(error)));
    await page.route('**/*', route => {
      const pathname = new URL(route.request().url()).pathname;
      if (pathname === '/api/v2/chat/sse') {
        const payload = route.request().postDataJSON();
        const body = ['run_started', 'text', 'done'].map((type, index) => `data: ${JSON.stringify({
          version: '2', sequence: index + 1, type,
          request_id: payload.request_id, conversation_id: payload.conversation_id,
          payload: type === 'text' ? { content: 'Fixture streamed reply' } : {},
        })}\n\n`).join('');
        return route.fulfill({ contentType: 'text/event-stream', body });
      }
      return pathname.startsWith('/api/') ? route.fulfill({ json: { enabled: false } }) : route.continue();
    });
    await page.addInitScript(() => {
      navigator.geolocation.getCurrentPosition = (_success, failure) => failure?.({ code: 1, message: 'fixture' });
      const sessions = [6, 2, 4, 0, 4].map((length, i) => ({
        id: `fixture-${i}`,
        chatId: `chat-fixture-${i}`,
        title: `Fixture ${i}`,
        date: '2026-10-13',
        messages: Array.from({ length }, (_, j) => ({
          role: j % 2 ? 'assistant' : 'user',
          content: j === 0 ? `Fixture ${i}` : `Session ${i} message ${j}`,
          ...(i === 4 && j % 2 ? {
            traceEvents: [{ type: 'tool_result', content: 'Fixture retrieval', metadata: {
              evidence: { has_chunks: true, chunks: [{ source: 'fixture', preview: 'Fixture evidence' }] },
            } }],
            artifacts: [{ artifact_id: `fixture-artifact-${j}`, public_filename: 'fixture.pdf', mime_type: 'application/pdf', size_bytes: 10 }],
          } : {}),
        })),
      }));
      localStorage.setItem('ai-hiking-agent-sessions', JSON.stringify(sessions));
      localStorage.setItem('ai-hiking-agent-active-session', sessions[0].id);
      localStorage.setItem('ai-hiking-agent-chat-id', sessions[0].chatId);
      localStorage.setItem('ai-hiking-agent-chat', JSON.stringify(sessions[0].messages));
    });
    await page.goto(process.env.BASE_URL || 'http://localhost:5173');
    await page.getByText('Session 0 message 5', { exact: true }).waitFor();
    const sequence = [1, 0, 2, 1, 3, 0, 4, 2, 2, 0];
    for (const i of Array.from({ length: 10 }, () => sequence).flat()) {
      await page.getByRole('complementary', { name: '对话历史' }).getByText(`Fixture ${i}`, { exact: true }).click();
      if (i === 3) {
        await page.getByRole('textbox', { name: '输入徒步问题' }).waitFor();
        assert.equal(await page.locator('[data-full-text]').count(), 0);
      } else {
        const lastIndex = [6, 2, 4, 0, 4][i] - 1;
        await page.locator(`[data-full-text="Session ${i} message ${lastIndex}"]`).waitFor();
        const content = await page.locator('[data-full-text]').evaluateAll(nodes => nodes.map(node => node.getAttribute('data-full-text')));
        assert.equal(content.length, lastIndex + 1);
        assert(content.every(text => text === `Fixture ${i}` || text.startsWith(`Session ${i} message `)));
      }
      assert.equal(await page.locator('#root > *').count(), 1, `root unmounted for session ${i}`);
      assert.deepEqual(errors, [], 'uncaught browser errors');
    }
    const sidebar = page.getByRole('complementary', { name: '对话历史' });
    await sidebar.getByRole('button', { name: '新建对话' }).click();
    await page.getByRole('textbox', { name: '输入徒步问题' }).waitFor();
    assert.equal(await page.locator('[data-full-text]').count(), 0);
    await sidebar.getByText('Fixture 0', { exact: true }).click();
    await page.locator('[data-full-text="Session 0 message 5"]').waitFor();
    await sidebar.getByText('Fixture 0', { exact: true }).locator('..').locator('..').getByRole('button', { name: '删除对话' }).click();
    await page.getByRole('textbox', { name: '输入徒步问题' }).waitFor();
    assert.equal(await page.locator('[data-full-text]').count(), 0);
    await sidebar.getByText('Fixture 4', { exact: true }).click();
    await page.locator('[data-full-text="Session 4 message 3"]').waitFor();
    await page.getByRole('textbox', { name: '输入徒步问题' }).fill('Fixture followup');
    await page.getByRole('button', { name: '发送消息' }).click();
    await page.locator('[data-full-text="Fixture streamed reply"]').waitFor();
    assert.equal(await page.locator('[data-full-text]').count(), 6);
    assert.equal(await page.locator('#root > *').count(), 1);
    assert.deepEqual(errors, []);
    if (process.env.SCREENSHOT_PATH) {
      await page.waitForTimeout(300); // Let the existing 220ms message entrance finish.
      await page.screenshot({ path: process.env.SCREENSHOT_PATH });
    }
    console.log('PASS: 100 session switches, exact content, empty/equal-length/tool sessions, new session, active deletion and send after switching; zero page errors.');
    } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); console.error(errors.join('\n')); process.exitCode = 1; });
