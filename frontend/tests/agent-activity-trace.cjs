// Browser seam for the inline Agent Activity Trace.
// Seeds localStorage sessions, loads the app through Vite, and mocks every
// /api route. A persisted assistant message with isStreaming: true maps to a
// running message status, so both running and completed states are covered
// without a live SSE server.
//
// Run with the dev server listening on BASE_URL:
//   BASE_URL=http://localhost:5177 node tests/agent-activity-trace.cjs
const { chromium } = require('playwright');
const assert = require('node:assert/strict');

const BASE_URL = process.env.BASE_URL || 'http://localhost:5173';

const event = (type, metadata = {}, content = '') => ({ type, metadata, content });
const searchCall = event('tool_call', { tool: 'hiking_knowledge_search' });
const searchOk = event('tool_result', {
  tool: 'hiking_knowledge_search',
  status: 'ok',
  duration_ms: 1200,
});
const searchError = event('tool_result', {
  tool: 'hiking_knowledge_search',
  status: 'error',
  duration_ms: 800,
  message: 'fixture failure',
});
const pdfApproval = event('approval_required', {
  approval_id: 'approval-fixture-1',
  tool: 'generate_pdf',
  message: '需要您的批准才能继续生成 PDF 并导出。',
});
const pdfApproved = event('approval_resolved', {
  approval_id: 'approval-fixture-1',
  tool: 'generate_pdf',
  decision: 'approve',
  status: 'approved',
  message: '用户已批准操作，继续执行。',
});
const pdfRejected = event('approval_resolved', {
  approval_id: 'approval-fixture-1',
  tool: 'generate_pdf',
  decision: 'reject',
  status: 'rejected',
});
const pdfResult = event('tool_result', {
  tool: 'generate_pdf',
  status: 'ok',
  duration_ms: 2400,
});

// assistant message indices in convertMessage map to data-message-id="msg-N".
const messages = [
  { role: 'user', content: 'Case 1 prompt' },
  { role: 'assistant', content: '', isStreaming: true },
  { role: 'user', content: 'Case 2 prompt' },
  {
    role: 'assistant',
    content: '',
    isStreaming: true,
    traceEvents: [searchCall, searchOk],
  },
  { role: 'user', content: 'Case 3 prompt' },
  { role: 'assistant', content: 'Case 3 answer', traceEvents: [searchOk] },
  { role: 'user', content: 'Case 4 prompt' },
  { role: 'assistant', content: 'Case 4 answer', traceEvents: [searchError] },
  { role: 'user', content: 'Case 5 prompt' },
  { role: 'assistant', content: 'Case 5 answer', traceEvents: [pdfApproval] },
  { role: 'user', content: 'Case 6 prompt' },
  {
    role: 'assistant',
    content: 'Case 6 answer',
    traceEvents: [pdfApproval, pdfApproved, pdfResult],
  },
  { role: 'user', content: 'Case 7 prompt' },
  {
    role: 'assistant',
    content: 'Case 7 answer',
    traceEvents: [pdfApproval, pdfRejected],
  },
  { role: 'user', content: 'Case 8 prompt' },
  {
    role: 'assistant',
    content: 'Case 8 answer',
    traceEvents: [event('future_event_type', { payload: 'ignored' })],
  },
  { role: 'user', content: 'Case 9 prompt' },
  {
    role: 'assistant',
    content: 'Case 9 answer',
    isStreaming: true,
    traceEvents: [searchOk],
  },
  { role: 'user', content: 'Case 10 prompt' },
  { role: 'assistant', content: 'Case 10 answer' },
];

let passed = 0;
function ok(name) {
  passed += 1;
  console.log(`PASS ${passed}: ${name}`);
}

async function launch(channelSetup = {}) {
  const browser = await chromium.launch({
    headless: true,
    channel: process.env.CHROME_PATH ? undefined : 'chrome',
    executablePath: process.env.CHROME_PATH,
  });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    ...channelSetup,
  });
  const page = await context.newPage();
  page.setDefaultTimeout(8000);
  const errors = [];
  page.on('pageerror', error => errors.push(error.stack || String(error)));
  page.on('console', message => {
    if (message.type() === 'error') errors.push(message.text());
  });
  await page.route('**/*', route => {
    const url = new URL(route.request().url());
    if (url.pathname.startsWith('/api/')) {
      return route.fulfill({ json: { enabled: false } });
    }
    // The app imports a Google Font; keep the suite independent of external
    // network access and of the load-event delay when that host is unreachable.
    if (url.hostname !== 'localhost' && url.hostname !== '127.0.0.1') {
      return route.fulfill({ contentType: 'text/css', body: '' });
    }
    return route.continue();
  });
  await page.addInitScript(() => {
    navigator.geolocation.getCurrentPosition = (_success, failure) =>
      failure?.({ code: 1, message: 'fixture' });
  });
  return { browser, page, errors };
}

async function seed(page) {
  await page.addInitScript(sessionMessages => {
    const sessions = [{
      id: 'trace-fixture-session',
      chatId: 'trace-fixture-chat',
      title: 'Agent Activity Fixture',
      date: '2026-10-24',
      messages: sessionMessages,
    }];
    localStorage.setItem('ai-hiking-agent-sessions', JSON.stringify(sessions));
    localStorage.setItem('ai-hiking-agent-active-session', sessions[0].id);
    localStorage.setItem('ai-hiking-agent-chat-id', sessions[0].chatId);
    localStorage.setItem('ai-hiking-agent-chat', JSON.stringify(sessions[0].messages));
  }, messages);
}

const messageRoot = (page, index) => page.locator(`[data-message-id="msg-${index}"]`);
const traceRoot = (page, index) => messageRoot(page, index).locator('[data-slot="agent-activity"]');
const contentLines = (page, index) =>
  traceRoot(page, index).locator('[data-slot="trace-collapsible-content"] [data-slot="trace-line"]');

(async () => {
  const { browser, page, errors } = await launch({ colorScheme: 'light' });
  try {
    await seed(page);
    await page.goto(BASE_URL);
    await page.getByText('Case 10 answer', { exact: true }).waitFor();

    // 1. Running, no events, empty content.
    assert.equal(await traceRoot(page, 1).count(), 1, 'case 1 trace visible');
    assert.equal(await traceRoot(page, 1).getByText('正在思考', { exact: true }).count(), 1);
    assert.equal(
      await traceRoot(page, 1).getByText('正在理解你的问题与上下文', { exact: true }).count(),
      1,
    );
    assert.equal(await contentLines(page, 1).count(), 1);
    ok('running without events shows thinking summary and contextual line');

    // 2. Running tool call closed by its result: one merged line.
    assert.equal(
      await traceRoot(page, 3).getByText('正在处理', { exact: true }).count(),
      1,
    );
    assert.equal(await contentLines(page, 3).count(), 1);
    assert.equal(
      await traceRoot(page, 3).getByText('已检索徒步知识库', { exact: true }).count(),
      1,
    );
    assert.equal(
      await traceRoot(page, 3).getByText('正在检索徒步知识库', { exact: true }).count(),
      0,
    );
    ok('tool_call + tool_result merges into one completed line');

    // 3. Completed result-only run: collapsed summary, expandable.
    const case3 = traceRoot(page, 5);
    const case3Trigger = case3.getByRole('button', { name: '已完成 1 个步骤', exact: true });
    assert.equal(await case3Trigger.count(), 1);
    assert.equal(await case3Trigger.getAttribute('aria-expanded'), 'false');
    assert.equal(
      await case3.getByText('已检索徒步知识库', { exact: true }).count(),
      0,
      'collapsed by default',
    );
    await case3Trigger.focus();
    await page.keyboard.press('Enter');
    assert.equal(await case3Trigger.getAttribute('aria-expanded'), 'true');
    await case3.getByText('已检索徒步知识库', { exact: true }).waitFor();
    assert.equal(await case3.getByText('1.2s', { exact: true }).count(), 1);
    await page.keyboard.press('Space');
    assert.equal(await case3Trigger.getAttribute('aria-expanded'), 'false');
    await case3.getByText('已检索徒步知识库', { exact: true }).waitFor({ state: 'detached' });
    assert.equal(
      await case3.getByText('已检索徒步知识库', { exact: true }).count(),
      0,
    );
    ok('clean completion collapses; keyboard toggles expand and collapse');

    // 4. Completed error result: expanded with the failure line.
    const case4 = traceRoot(page, 7);
    assert.equal(await case4.getByText('有步骤执行失败', { exact: true }).count(), 1);
    assert.equal(
      await case4.getByRole('button', { name: '有步骤执行失败', exact: true })
        .getAttribute('aria-expanded'),
      'true',
    );
    assert.equal(
      await case4.getByText('检索徒步知识库失败', { exact: true }).count(),
      1,
    );
    ok('failure stays expanded and shows the error line');

    // 5. Approval wait: pending line, no approve/reject controls.
    const case5 = traceRoot(page, 9);
    assert.equal(await case5.getByText('等待你的确认', { exact: true }).count(), 1);
    assert.equal(
      await case5.getByText('等待确认：生成计划书 PDF', { exact: true }).count(),
      1,
    );
    assert.equal(await case5.getByRole('button', { name: /^(确认|拒绝)$/ }).count(), 0);
    assert.equal(await case5.getByRole('button').count(), 1, 'only the summary trigger');
    ok('approval wait shows pending line without approve or reject buttons');

    // 6. Approved approval settles the line; summary leaves the wait state.
    const case6 = traceRoot(page, 11);
    const case6Trigger = case6.getByRole('button', { name: '已完成 1 个步骤', exact: true });
    assert.equal(await case6Trigger.count(), 1);
    assert.equal(await case6.getByText('等待你的确认', { exact: true }).count(), 0);
    assert.equal(await case6Trigger.getAttribute('aria-expanded'), 'false');
    await case6Trigger.click();
    await case6.getByText('已确认：生成计划书 PDF', { exact: true }).waitFor();
    assert.equal(
      await case6.getByText('已确认：生成计划书 PDF', { exact: true }).count(),
      1,
    );
    ok('approve resolves the pending approval line into a done line');

    // 7. Rejected approval: error line and rejected summary, expanded.
    const case7 = traceRoot(page, 13);
    assert.equal(await case7.getByText('操作已拒绝', { exact: true }).count(), 1);
    assert.equal(
      await case7.getByRole('button', { name: '操作已拒绝', exact: true })
        .getAttribute('aria-expanded'),
      'true',
    );
    assert.equal(
      await case7.getByText('已拒绝：生成计划书 PDF', { exact: true }).count(),
      1,
    );
    ok('reject resolves the approval line into an error line and keeps it expanded');

    // 8. Unknown event only: no trace at all.
    assert.equal(await messageRoot(page, 15).getByText('Case 8 answer', { exact: true }).count(), 1);
    assert.equal(await traceRoot(page, 15).count(), 0);
    ok('unknown event types produce no trace');

    // 9. Running with answer text: generating summary and answer line.
    const case9 = traceRoot(page, 17);
    assert.equal(await case9.getByText('正在生成回答', { exact: true }).count(), 1);
    assert.equal(await case9.getByText('正在整理答案', { exact: true }).count(), 1);
    assert.equal(await case9.getByText('已检索徒步知识库', { exact: true }).count(), 1);
    assert.equal(await contentLines(page, 17).count(), 2);
    ok('running with answer text shows the generating summary and answer line');

    // 10. Completed without events: no trace.
    assert.equal(await traceRoot(page, 19).count(), 0);
    assert.equal(await page.locator('[data-slot="agent-activity"]').count(), 8);
    ok('completed run without events produces no trace');

    // Old widgets are gone.
    assert.equal(await page.getByText('思考流程信息', { exact: false }).count(), 0);
    assert.equal(await page.getByText('模型思考中', { exact: false }).count(), 0);
    ok('old thinking widgets are absent');

    // Normal motion keeps the inherited shimmer on the running title.
    const shimmer = await traceRoot(page, 1)
      .locator('[data-slot="trace-title"]')
      .first()
      .evaluate(node => getComputedStyle(node).animationName);
    assert.match(shimmer, /shimmer/, 'running title keeps the project shimmer');
    ok('running title inherits the shimmer animation');

    assert.deepEqual(errors, [], 'no page or console errors');
    ok('light mode loads without page or console errors');
  } finally {
    await browser.close();
  }

  // Dark mode plus reduced motion.
  const dark = await launch({ colorScheme: 'dark', reducedMotion: 'reduce' });
  try {
    await seed(dark.page);
    await dark.page.goto(BASE_URL);
    await dark.page.getByText('Case 10 answer', { exact: true }).waitFor();
    const darkTrace = traceRoot(dark.page, 1);
    assert.equal(await darkTrace.count(), 1);
    assert.equal(await darkTrace.getByText('正在思考', { exact: true }).count(), 1);
    const reducedAnimation = await darkTrace
      .locator('[data-slot="trace-title"]')
      .first()
      .evaluate(node => getComputedStyle(node).animationName);
    assert.equal(reducedAnimation, 'none', 'reduced motion disables the shimmer');
    assert.equal(
      await traceRoot(dark.page, 7).getByText('检索徒步知识库失败', { exact: true }).count(),
      1,
    );
    await dark.page.waitForTimeout(250);
    assert.deepEqual(dark.errors, [], 'no page or console errors in dark mode');
    ok('dark mode renders the trace with no errors');
    ok('reduced motion disables the running-title shimmer');
  } finally {
    await dark.browser.close();
  }

  console.log(`\nAll ${passed} agent activity trace checks passed.`);
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
