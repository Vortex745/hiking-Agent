// Run against the Vite server with BASE_URL and optional CHROME_PATH.
// Seeds a legacy transcript (old backend appended raw MCP JSON after the
// 'MCP 返回：' marker) and asserts the display layer hides the JSON, renders
// the image URLs, and keeps plain-text tool output.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');

const STORAGE_KEY = 'ai-hiking-agent-chat';
const PREVIEW = 'https://preview.qiantucdn.com/legacy-test.jpg!qt_kuan320';

const modelList = {
  ok: true,
  data: { list: [{ id: 4821, title: '即梦5.0', max_generate_num: 4 }] },
};
const metaOnly = {
  ok: true,
  data: { balance_points: 128, tier: 'free' },
};
const imageList = {
  ok: true,
  data: { list: [{ title: '雪山徒步', preview_url: PREVIEW }] },
};
const plainJsonAnswer = '{"route": "龙脊线", "distance_km": 12}';

const messages = [
  { role: 'user', content: '给我几张图片' },
  {
    role: 'assistant',
    content: `参数确认好了。现在生成图片：\n\nMCP 返回：\n${JSON.stringify(modelList, null, 2)}\n\n${JSON.stringify(imageList, null, 2)}`,
  },
  { role: 'user', content: '另一个问题' },
  {
    role: 'assistant',
    content: '工具结果：\n\nMCP 返回：\necho:hello:walk:',
  },
  { role: 'user', content: '我的余额还有多少' },
  {
    role: 'assistant',
    content: `余额查询完成。\n\nMCP 返回：\n${JSON.stringify(metaOnly, null, 2)}`,
  },
  { role: 'user', content: '给我一段机读结果' },
  { role: 'assistant', content: `以下是原样 JSON：\n${plainJsonAnswer}` },
];

(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    page.setDefaultTimeout(15000);
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    await page.addInitScript(({ key, seed }) => {
      navigator.geolocation.getCurrentPosition = (_s, failure) => failure?.({ code: 1, message: 'fixture' });
      localStorage.setItem(key, JSON.stringify(seed));
      localStorage.setItem('ai-hiking-agent-chat-id', 'chat-legacy-fixture');
      localStorage.removeItem('ai-hiking-agent-v2-active-run');
      localStorage.removeItem('ai-hiking-agent-active-session');
    }, { key: STORAGE_KEY, seed: messages });
    await page.route('**/*', route => {
      // JWT 鉴权门 fixture：浏览器测试默认已登录
      if (new URL(route.request().url()).pathname === '/api/auth/me') return route.fulfill({ json: { user_id: 'test-user', login_id: 'tester', created_at: '2026-01-01T00:00:00Z' } });
      const pathname = new URL(route.request().url()).pathname;
      return pathname.startsWith('/api/') ? route.fulfill({ json: { enabled: false } }) : route.continue();
    });
    await page.goto(process.env.BASE_URL || 'http://127.0.0.1:5176/');

    await page.getByText('参数确认好了。现在生成图片：').waitFor();
    const texts = await page.locator('[data-full-text]').evaluateAll(nodes => nodes.map(n => n.getAttribute('data-full-text') || ''));
    const rendered = texts.join('\n');
    assert.ok(rendered.includes('参数确认好了。现在生成图片：'), 'model text preserved');
    assert.ok(!rendered.includes('"preview_url"'), 'legacy preview_url JSON hidden from the rendered text');
    assert.ok(!rendered.includes('"max_generate_num"'), 'legacy model-list JSON hidden from the rendered text');
    assert.ok(rendered.includes(`![雪山徒步](${PREVIEW})`), 'image URL extracted from legacy JSON as Markdown');
    assert.ok(rendered.includes('echo:hello:walk:'), 'plain-text MCP output kept');
    assert.ok(rendered.includes('余额查询完成。'), 'metadata-only legacy prefix kept');
    assert.ok(rendered.includes(plainJsonAnswer), 'ordinary JSON answer is not stripped');

    const bodyText = await page.locator('body').innerText();
    assert.ok(!bodyText.includes('"preview_url"') && !bodyText.includes('"max_generate_num"'), 'no legacy JSON visible on the page');
    assert.ok(!bodyText.includes('"balance_points"'), 'metadata-only legacy JSON hidden from the page');
    const images = page.locator(`img[src="${PREVIEW}"]`);
    assert.equal(await images.count(), 1, 'legacy image rendered once');

    assert.deepEqual(errors, [], 'uncaught browser errors');
    const loadedAssets = await page.evaluate(() => Array.from(document.querySelectorAll('script[src], link[rel="stylesheet"]'))
      .map(node => node.getAttribute('src') || node.getAttribute('href')));
    assert.ok(loadedAssets.length > 0, 'the page loaded its module assets');
    // Set EXPECT_BUILT_ASSETS=1 when BASE_URL serves the production preview;
    // the dev server legitimately serves /@vite/client and /src/main.tsx.
    if (process.env.EXPECT_BUILT_ASSETS === '1') {
      assert.ok(loadedAssets.every(asset => /\/assets\//.test(asset)), loadedAssets.join(','));
    }
    await page.screenshot({ path: '../.tmp/mcp-legacy-output.png' });
    console.log(`PASS: legacy MCP JSON hidden, image extracted, plain-text and ordinary JSON kept. Assets: ${loadedAssets.join(', ')}`);
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
