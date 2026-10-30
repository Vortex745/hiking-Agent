// Run against the Vite dev/preview server with BASE_URL and optional CHROME_PATH.
// Seeds single-image, 4-image and 6-image turns, then checks images stay in the
// normal message flow and open in Viewer.js with navigation and focus return.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');

const IMAGE = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAFElEQVR42mNk+M9Qz0AEYBxVSF+FABJADveWkH6oAAAAAElFTkSuQmCC',
  'base64',
);
const STORAGE_KEY = 'ai-hiking-agent-chat';
const imageUrl = (name) => `https://example.com/${name}.png`;
const imageLine = (name, index) => `![图${index + 1}](${imageUrl(name)})`;

const messages = [
  { role: 'user', content: '单图测试' },
  { role: 'assistant', content: `只有一张：\n\n![单图](${imageUrl('one')})` },
  { role: 'user', content: '四图测试' },
  {
    role: 'assistant',
    content: `四张图：\n\n${['one', 'two', 'three', 'four'].map(imageLine).join('\n')}`,
  },
  { role: 'user', content: '六图测试' },
  {
    role: 'assistant',
    content: `六张图：\n\n${['one', 'two', 'three', 'four', 'five', 'six'].map(imageLine).join('\n')}`,
  },
];

(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    page.setDefaultTimeout(15000);
    const errors = [];
    page.on('pageerror', (error) => errors.push(String(error)));
    await page.addInitScript(({ key, seed }) => {
      navigator.geolocation.getCurrentPosition = (_success, failure) => failure?.({ code: 1, message: 'fixture' });
      const session = { id: 'image-fixture', chatId: 'chat-image-fixture', title: '图片展示', date: '2026-10-28', messages: seed };
      localStorage.setItem('ai-hiking-agent-sessions', JSON.stringify([session]));
      localStorage.setItem('ai-hiking-agent-active-session', session.id);
      localStorage.setItem('ai-hiking-agent-chat-id', session.chatId);
      localStorage.setItem(key, JSON.stringify(seed));
      localStorage.removeItem('ai-hiking-agent-v2-active-run');
    }, { key: STORAGE_KEY, seed: messages });
    await page.route('**/*', (route) => {
      // JWT 鉴权门 fixture：浏览器测试默认已登录
      if (new URL(route.request().url()).pathname === '/api/auth/me') return route.fulfill({ json: { user_id: 'test-user', login_id: 'tester', created_at: '2026-01-01T00:00:00Z' } });
      const pathname = new URL(route.request().url()).pathname;
      return pathname.startsWith('/api/') ? route.fulfill({ json: { enabled: false } }) : route.continue();
    });
    // Register after the catch-all: Playwright runs the newest handler first.
    await page.route('https://example.com/*.png', (route) => route.fulfill({ contentType: 'image/png', body: IMAGE }));
    await page.goto(process.env.BASE_URL || 'http://127.0.0.1:5173/');
    await page.getByText('四张图：').waitFor();

    assert.equal(await page.locator('[data-chat-images="single"]').count(), 1, 'single image keeps its own block');
    assert.equal(await page.locator('[data-chat-images="multi"]').count(), 2, 'multi-image turns render their own blocks');

    const four = page.locator('[data-image-count="4"]');
    assert.equal(await four.getByRole('button', { name: /^放大图片：图\d$/ }).count(), 4, 'all four images are clickable');
    assert.equal(await four.locator('img').count(), 4, 'no duplicate image nodes');

    const bodyText = await page.locator('body').innerText();
    assert.ok(!bodyText.includes('![图2]') && !bodyText.includes('example.com/two.png'), 'images leave no raw Markdown behind');

    const six = page.locator('[data-image-count="6"]');
    assert.equal(await six.getByRole('button', { name: /^放大图片：图\d$/ }).count(), 6, 'all six images render');
    assert.equal(await six.locator('img').count(), 6, 'every gallery image has a node');
    assert.equal(await six.getByText('+1').count(), 0, 'no stack badge remains');

    // Images must sit in normal flow: distinct vertical positions, no overlap.
    const boxes = [];
    for (const card of await four.getByRole('button', { name: /^放大图片：图\d$/ }).all()) {
      boxes.push(await card.boundingBox());
    }
    for (let i = 1; i < boxes.length; i += 1) {
      assert.ok(boxes[i].y >= boxes[i - 1].y + boxes[i - 1].height, `image ${i + 1} flows below image ${i}`);
    }

    await four.getByRole('button', { name: '放大图片：图3' }).click();
    await page.locator('.viewer-container:not(.viewer-hide)').waitFor({ state: 'visible' });
    await page.waitForFunction(() => {
      const image = document.querySelector('.viewer-container:not(.viewer-hide) .viewer-canvas img');
      return Boolean(image && image.naturalWidth > 0 && image.src.includes('three.png'));
    });

    await page.keyboard.press('ArrowRight');
    await page.waitForFunction(() => {
      const image = document.querySelector('.viewer-container:not(.viewer-hide) .viewer-canvas img');
      return Boolean(image && image.naturalWidth > 0 && image.src.includes('four.png'));
    });

    await page.keyboard.press('Escape');
    await page.waitForFunction(() => !document.querySelector('.viewer-container:not(.viewer-hide)'));
    await page.waitForFunction(() => document.activeElement?.getAttribute('aria-label') === '放大图片：图3');

    await six.getByRole('button', { name: '放大图片：图5' }).click();
    await page.locator('.viewer-container:not(.viewer-hide)').waitFor({ state: 'visible' });
    await page.waitForFunction(() => {
      const image = document.querySelector('.viewer-container:not(.viewer-hide) .viewer-canvas img');
      return Boolean(image && image.naturalWidth > 0 && image.src.includes('five.png'));
    });
    await page.keyboard.press('ArrowRight');
    await page.waitForFunction(() => {
      const image = document.querySelector('.viewer-container:not(.viewer-hide) .viewer-canvas img');
      return Boolean(image && image.naturalWidth > 0 && image.src.includes('six.png'));
    });
    await page.keyboard.press('Escape');
    await page.waitForFunction(() => !document.querySelector('.viewer-container:not(.viewer-hide)'));
    await page.waitForFunction(() => document.activeElement?.getAttribute('aria-label') === '放大图片：图5');

    const single = page.locator('[data-chat-images="single"]');
    await single.getByRole('button', { name: '放大图片：单图' }).click();
    await page.locator('.viewer-container:not(.viewer-hide)').waitFor({ state: 'visible' });
    await page.waitForFunction(() => {
      const image = document.querySelector('.viewer-container:not(.viewer-hide) .viewer-canvas img');
      return Boolean(image && image.naturalWidth > 0 && image.src.includes('one.png'));
    });
    await page.keyboard.press('Escape');
    await page.waitForFunction(() => !document.querySelector('.viewer-container:not(.viewer-hide)'));

    await page.screenshot({ path: '../.tmp/chat-images-1280.png' });

    await page.setViewportSize({ width: 390, height: 844 });
    for (const block of await page.locator('[data-chat-images]').all()) {
      const box = await block.boundingBox();
      assert.ok(box && box.x >= 0 && box.x + box.width <= 390, `images stay inside 390px, got ${JSON.stringify(box)}`);
    }
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    assert.ok(overflow <= 0, `no horizontal overflow at 390px, got ${overflow}px`);
    await page.screenshot({ path: '../.tmp/chat-images-390.png' });

    assert.deepEqual(errors, [], 'uncaught browser errors');
    console.log('PASS: chat images (1/4/6), normal flow, Viewer.js navigation, focus return and 390px bounds.');
  } finally {
    await browser.close();
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
