// MCP display acceptance: one asset renders one preview, a normal link stays a
// link, discovery/authorization rows never count as execution steps, and an
// unknown call is shown as unconfirmed (never success or definite failure).
//
// Run against the Vite dev/preview server:
//   BASE_URL=http://127.0.0.1:5179 CHROME_PATH=... node tests/mcp-display.cjs
const { chromium } = require('playwright');
const assert = require('node:assert/strict');

const IMAGE = Buffer.from(
  `<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360">
    <rect width="640" height="360" fill="#dbeafe"/>
    <path d="M0 300 L190 120 L330 260 L450 170 L640 310 L640 360 L0 360 Z" fill="#60a5fa"/>
    <circle cx="520" cy="80" r="34" fill="#fbbf24"/>
    <text x="24" y="52" font-size="30" fill="#1e3a8a">白云山</text>
  </svg>`,
);
const STORAGE_KEY = 'ai-hiking-agent-chat';
const PREVIEW = 'https://preview.example.com/white-cloud.jpg!qt_kuan320';
const ORIGIN = 'https://origin.example.com/white-cloud.jpg?token=protected';
const EXTERNAL = 'https://example.com/page';

const event = (type, metadata = {}, content = '') => ({ type, metadata, content });
const discovery = event('tool_result', {
  tool: 'mcp__svc__discovery',
  status: 'ok',
  source: 'mcp',
  server_id: 'svc',
});
const authorized = event('tool_result', {
  tool: 'mcp__svc__echo',
  status: 'authorized',
  source: 'mcp',
  server_id: 'svc',
});
const callOk = index => event('tool_result', {
  tool: 'mcp__svc__echo',
  status: 'ok',
  duration_ms: 100 + index,
  source: 'mcp',
  server_id: 'svc',
});
const callFailed = event('tool_result', {
  tool: 'mcp__svc__generate_image',
  status: 'error',
  duration_ms: 900,
  source: 'mcp',
  server_id: 'svc',
});
const callUnknown = event('tool_result', {
  tool: 'mcp__svc__generate_image',
  status: 'outcome_unknown',
  source: 'mcp',
  server_id: 'svc',
  content: '本次调用结果未确认',
});

const messages = [
  { role: 'user', content: '图片与链接' },
  {
    role: 'assistant',
    content: [
      `预览：![白云山](${PREVIEW})`,
      `原图：[原图](${ORIGIN})`,
      `相关：[官网](${EXTERNAL})`,
    ].join('\n\n'),
  },
  { role: 'user', content: '执行轨迹' },
  {
    role: 'assistant',
    content: '六步调用完成。',
    traceEvents: [
      discovery,
      authorized,
      ...Array.from({ length: 6 }, (_, index) => callOk(index)),
    ],
  },
  { role: 'user', content: '未知调用' },
  {
    role: 'assistant',
    content: '结果未确认。',
    traceEvents: [discovery, callUnknown],
  },
  { role: 'user', content: '失败与未知' },
  {
    role: 'assistant',
    content: '部分失败。',
    traceEvents: [callFailed, callUnknown],
  },
];

(async () => {
  const browser = await chromium.launch({
    headless: true,
    channel: process.env.CHROME_PATH ? undefined : 'chrome',
    executablePath: process.env.CHROME_PATH,
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    page.setDefaultTimeout(15000);
    const errors = [];
    const requests = [];
    page.on('pageerror', error => errors.push(String(error)));
    page.on('console', message => {
      if (message.type() === 'error') errors.push(message.text());
    });
    page.on('request', request => requests.push(request.url()));

    await page.addInitScript(({ key, seed }) => {
      navigator.geolocation.getCurrentPosition = (_success, failure) =>
        failure?.({ code: 1, message: 'fixture' });
      const session = {
        id: 'mcp-display-fixture',
        chatId: 'chat-mcp-display',
        title: 'MCP 展示',
        date: '2026-10-28',
        messages: seed,
      };
      localStorage.setItem('ai-hiking-agent-sessions', JSON.stringify([session]));
      localStorage.setItem('ai-hiking-agent-active-session', session.id);
      localStorage.setItem('ai-hiking-agent-chat-id', session.chatId);
      localStorage.setItem(key, JSON.stringify(seed));
      localStorage.removeItem('ai-hiking-agent-v2-active-run');
    }, { key: STORAGE_KEY, seed: messages });

    await page.route('**/*', route => {
      const url = new URL(route.request().url());
      if (url.pathname.startsWith('/api/')) {
        return route.fulfill({ json: { enabled: false } });
      }
      if (url.hostname === 'preview.example.com') {
        return route.fulfill({ contentType: 'image/svg+xml', body: IMAGE });
      }
      if (url.hostname !== 'localhost' && url.hostname !== '127.0.0.1') {
        return route.fulfill({ contentType: 'text/css', body: '' });
      }
      return route.continue();
    });

    await page.goto(process.env.BASE_URL || 'http://127.0.0.1:5179/', {
      waitUntil: 'domcontentloaded',
    });
    await page.getByText('六步调用完成。', { exact: true }).waitFor();

    const messageRoot = index => page.locator(`[data-message-id="msg-${index}"]`);
    const traceRoot = index =>
      messageRoot(index).locator('[data-slot="agent-activity"]');

    // 1. One image syntax -> one preview; ordinary links stay links.
    const imageMessage = messageRoot(1);
    assert.equal(await imageMessage.locator('[data-chat-images="single"]').count(), 1);
    assert.equal(await imageMessage.locator('[data-chat-images] img').count(), 1);
    assert.equal(
      await imageMessage.locator(`img[src="${PREVIEW}"]`).count(),
      1,
      'the preview image renders once',
    );
    assert.equal(
      await imageMessage.locator('img').count(),
      1,
      'a plain .jpg link must not be upgraded into a second image',
    );
    const originLink = imageMessage.getByRole('link', { name: '原图', exact: true });
    assert.equal(await originLink.count(), 1);
    assert.equal(await originLink.getAttribute('href'), ORIGIN);
    assert.equal(await originLink.getAttribute('target'), '_blank');
    assert.equal(
      await imageMessage.getByRole('link', { name: '官网', exact: true }).count(),
      1,
    );
    const bodyText = await imageMessage.innerText();
    assert.ok(!bodyText.includes('![白云山]'), 'image Markdown leaves no raw text');
    assert.ok(!bodyText.includes('共 1 张图片'), 'no automatic image-count tail');
    assert.ok(
      !requests.some(url => url.startsWith('https://origin.example.com')),
      'a protected original is never requested before the user clicks it',
    );
    await imageMessage.scrollIntoViewIfNeeded();
    await page.screenshot({ path: '../.tmp/mcp-display-links-1280.png' });

    await imageMessage.getByRole('button', { name: '放大图片：白云山' }).click();
    await page.locator('.viewer-container:not(.viewer-hide)').waitFor({ state: 'visible' });
    await page.waitForFunction(() => {
      const image = document.querySelector('.viewer-container:not(.viewer-hide) .viewer-canvas img');
      return Boolean(image && image.src.includes('white-cloud.jpg'));
    });
    await page.keyboard.press('Escape');
    await page.waitForFunction(() => !document.querySelector('.viewer-container:not(.viewer-hide)'));
    assert.equal(await originLink.getAttribute('href'), ORIGIN, 'viewer does not rewrite the link');
    console.log('PASS: one preview per object, normal links stay links, viewer works');

    // 2. Discovery and authorization rows are excluded from steps.
    const traceMessage = traceRoot(3);
    const traceTrigger = traceMessage.getByRole('button', {
      name: '已完成 6 个步骤',
      exact: true,
    });
    assert.equal(await traceTrigger.count(), 1);
    await traceTrigger.click();
    const lines = traceMessage.locator(
      '[data-slot="trace-collapsible-content"] [data-slot="trace-line"]',
    );
    assert.equal(await lines.count(), 6, 'six real executions keep six result lines');
    assert.equal(
      await traceMessage.getByText(/discovery|授权|authorized/i).count(),
      0,
      'discovery/authorization never appear as execution',
    );
    console.log('PASS: discovery and authorization are excluded from execution steps');

    // 3. Unknown-only summary is truthful and does not claim success.
    const unknownTrace = traceRoot(5);
    assert.equal(
      await unknownTrace.getByText('有调用结果未确认', { exact: true }).count(),
      1,
    );
    assert.equal(
      await unknownTrace.getByText('本次调用结果未确认', { exact: true }).count(),
      1,
    );
    assert.equal(
      await unknownTrace.getByRole('button', { name: /已完成/ }).count(),
      0,
      'unknown is not reported as a completed step count',
    );
    console.log('PASS: an unknown call shows the unconfirmed summary and line');

    // 4. Failure + unknown: neither is hidden by the other.
    const mixedTrace = traceRoot(7);
    assert.equal(
      await mixedTrace.getByText('有调用失败或未确认', { exact: true }).count(),
      1,
    );
    assert.equal(
      await mixedTrace.getByText('本次调用结果未确认', { exact: true }).count(),
      1,
    );
    assert.equal(
      await mixedTrace.getByText(/generate_image失败/).count(),
      1,
    );
    console.log('PASS: failure + unknown keeps both states visible');

    // 5. Reconnect/replay consistency: the seeded persisted payload renders
    // identically after a reload.
    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.getByText('六步调用完成。', { exact: true }).waitFor();
    assert.equal(await messageRoot(1).locator('[data-chat-images] img').count(), 1);
    assert.equal(
      await messageRoot(1).getByRole('link', { name: '原图', exact: true }).count(),
      1,
    );
    assert.equal(
      await traceRoot(5).getByText('有调用结果未确认', { exact: true }).count(),
      1,
    );
    console.log('PASS: persisted replay renders the same image/link/trace state');

    await page.screenshot({ path: '../.tmp/mcp-display-1280.png' });

    // 6. 390px: links stay clickable and nothing overflows.
    await page.setViewportSize({ width: 390, height: 844 });
    await imageMessage.scrollIntoViewIfNeeded();
    const imageBox = await imageMessage.locator('[data-chat-images]').boundingBox();
    assert.ok(
      imageBox && imageBox.x >= 0 && imageBox.x + imageBox.width <= 390,
      `preview stays inside 390px, got ${JSON.stringify(imageBox)}`,
    );
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - window.innerWidth,
    );
    assert.ok(overflow <= 0, `no horizontal overflow at 390px, got ${overflow}px`);
    assert.equal(
      await imageMessage.getByRole('link', { name: '原图', exact: true }).isVisible(),
      true,
    );
    await page.screenshot({ path: '../.tmp/mcp-display-390.png' });

    assert.deepEqual(errors, [], 'no page or console errors');
    console.log('PASS: 390px bounds and console-clean render');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
