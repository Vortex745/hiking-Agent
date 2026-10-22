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
const searchCall = event('tool_call', {
  tool: 'hiking_knowledge_search',
  args: { query: '雪线' },
});
const searchOk = event('tool_result', {
  tool: 'hiking_knowledge_search',
  status: 'ok',
  duration_ms: 1200,
  evidence: { has_chunks: true, chunks: [{}, {}, {}, {}] },
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
const pdfExpired = event('approval_resolved', {
  approval_id: 'approval-fixture-1',
  tool: 'generate_pdf',
  status: 'expired',
  message: '确认已超时。',
});
const pdfResult = event('tool_result', {
  tool: 'generate_pdf',
  status: 'ok',
  duration_ms: 2400,
});
const parallelCallA = event('tool_call', { tool: 'hiking_knowledge_search' });
const parallelCallB = event('tool_call', { tool: 'hiking_knowledge_search' });
const parallelResultA = event('tool_result', {
  tool: 'hiking_knowledge_search',
  status: 'ok',
  duration_ms: 1100,
});
const parallelResultB = event('tool_result', {
  tool: 'hiking_knowledge_search',
  status: 'ok',
  duration_ms: 2300,
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
  { role: 'user', content: 'Case 11 prompt' },
  {
    // Real paused state: run.paused leaves the message streaming.
    role: 'assistant',
    content: '',
    isStreaming: true,
    traceEvents: [pdfApproval],
  },
  { role: 'user', content: 'Case 12 prompt' },
  {
    role: 'assistant',
    content: 'Case 12 answer',
    traceEvents: [pdfApproval, pdfExpired],
  },
  { role: 'user', content: 'Case 13 prompt' },
  {
    role: 'assistant',
    content: 'Case 13 answer',
    traceEvents: [parallelCallA, parallelCallB, parallelResultA, parallelResultB],
  },
  { role: 'user', content: 'Case 14 prompt' },
  {
    role: 'assistant',
    content: 'Case 14 answer',
    traceEvents: [
      event('thought', { step: 0 }, '先确认用户问的是雪线徒步，需要结合季节与海拔判断风险。'),
      event('thought', { step: 0 }, '检索知识库中的雪线注意事项，再给出保守建议。'),
    ],
  },
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
  page.setDefaultTimeout(20000);
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
    await page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    await page.getByText('Case 10 answer', { exact: true }).waitFor();

    // 1. Running, no events, empty content.
    assert.equal(await traceRoot(page, 1).count(), 1, 'case 1 trace visible');
    assert.equal(await traceRoot(page, 1).getByText('正在思考', { exact: true }).count(), 1);
    assert.equal(
      await traceRoot(page, 1).getByText('正在理解你的问题与上下文', { exact: true }).count(),
      1,
    );
    assert.equal(await contentLines(page, 1).count(), 1);
    // Only the executing step rotates; the summary icon stays static.
    const summarySpins = await traceRoot(page, 1)
      .locator('[data-slot="trace-collapsible-trigger"] [data-slot="trace-icon"] svg')
      .first()
      .evaluate(node => node.classList.contains('animate-spin'));
    assert.equal(summarySpins, false, 'summary icon must not rotate');
    const stepSpins = await contentLines(page, 1)
      .first()
      .locator('[data-slot="trace-icon"] svg')
      .first()
      .evaluate(node => node.classList.contains('animate-spin'));
    assert.equal(stepSpins, true, 'the executing step keeps the rotating icon');
    ok('running without events rotates only the executing step');

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
    assert.equal(
      await traceRoot(page, 3)
        .getByText('查询「雪线」 · 1.2s · 4 条', { exact: true })
        .count(),
      1,
      'call arguments and result detail share one description',
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
    const blockColor = await case3
      .locator('[data-slot="agent-activity-block"]')
      .evaluate(node => getComputedStyle(node).backgroundColor);
    assert.notEqual(blockColor, 'rgba(0, 0, 0, 0)', 'trace sits on a filled block');
    await case3Trigger.focus();
    await page.keyboard.press('Enter');
    assert.equal(await case3Trigger.getAttribute('aria-expanded'), 'true');
    await case3.getByText('已检索徒步知识库', { exact: true }).waitFor();
    assert.equal(await case3.getByText('1.2s · 4 条', { exact: true }).count(), 1);
    const openAnimation = await case3
      .locator('[data-slot="trace-collapsible-content"]')
      .evaluate(node => getComputedStyle(node).animationName);
    assert.match(
      openAnimation,
      /trace-collapse-down/,
      'user expansion animates after mount',
    );
    ok('user expansion runs the collapse animation');
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
    const mountAnimation = await case4
      .locator('[data-slot="trace-collapsible-content"]')
      .evaluate(node => getComputedStyle(node).animationName);
    assert.equal(
      mountAnimation,
      'none',
      'an already-expanded trace must not animate on initial mount',
    );
    ok('initially expanded trace skips the mount animation');

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

    // 8. Unknown event only: no activity line, just the settled record.
    assert.equal(await messageRoot(page, 15).getByText('Case 8 answer', { exact: true }).count(), 1);
    const case8 = traceRoot(page, 15);
    const case8Trigger = case8.getByRole('button', { name: '思考过程', exact: true });
    assert.equal(await case8Trigger.count(), 1);
    await case8Trigger.click();
    assert.equal(await contentLines(page, 15).count(), 1);
    assert.equal(
      await case8.getByText('已理解你的问题并生成回答', { exact: true }).count(),
      1,
    );
    ok('unknown event types produce no activity line');

    // 9. Running with answer text: generating summary and answer line.
    const case9 = traceRoot(page, 17);
    assert.equal(await case9.getByText('正在生成回答', { exact: true }).count(), 1);
    assert.equal(await case9.getByText('正在整理答案', { exact: true }).count(), 1);
    assert.equal(await case9.getByText('已检索徒步知识库', { exact: true }).count(), 1);
    assert.equal(await contentLines(page, 17).count(), 2);
    ok('running with answer text shows the generating summary and answer line');

    // 10. Completed without events keeps a collapsed thinking record.
    const case10 = traceRoot(page, 19);
    const case10Trigger = case10.getByRole('button', { name: '思考过程', exact: true });
    assert.equal(await case10Trigger.count(), 1);
    assert.equal(await case10Trigger.getAttribute('aria-expanded'), 'false');
    await case10Trigger.click();
    await case10.getByText('已理解你的问题并生成回答', { exact: true }).waitFor();
    assert.equal(
      await case10.getByText('已理解你的问题并生成回答', { exact: true }).count(),
      1,
    );
    assert.equal(await page.locator('[data-slot="agent-activity"]').count(), 14);
    ok('completed run without events keeps a collapsed thinking record');

    // 11. Paused approval while the run is still streaming: the wait state
    // must win even though isStreaming keeps the message running.
    const case11 = traceRoot(page, 21);
    assert.equal(await case11.getByText('等待你的确认', { exact: true }).count(), 1);
    assert.equal(
      await case11.getByRole('button', { name: '等待你的确认', exact: true })
        .getAttribute('aria-expanded'),
      'true',
    );
    assert.equal(
      await case11.getByText('等待确认：生成计划书 PDF', { exact: true }).count(),
      1,
    );
    assert.equal(await contentLines(page, 21).count(), 1);
    ok('paused streaming approval shows the wait summary and pending line');

    // 12. Expired approval resolves the wait into an error line.
    const case12 = traceRoot(page, 23);
    assert.equal(await case12.getByText('确认已过期', { exact: true }).count(), 1);
    assert.equal(
      await case12.getByRole('button', { name: '确认已过期', exact: true })
        .getAttribute('aria-expanded'),
      'true',
    );
    assert.equal(
      await case12.getByText('已过期：生成计划书 PDF', { exact: true }).count(),
      1,
    );
    assert.equal(await case12.getByText('确认已超时。', { exact: true }).count(), 1);
    ok('expired approval becomes an error line with the expired summary');

    // 13. Parallel same-tool calls pair FIFO with their own results.
    const case13 = traceRoot(page, 25);
    const case13Trigger = case13.getByRole('button', {
      name: '已完成 2 个步骤',
      exact: true,
    });
    assert.equal(await case13Trigger.count(), 1);
    assert.equal(await case13Trigger.getAttribute('aria-expanded'), 'false');
    await case13Trigger.click();
    assert.equal(await contentLines(page, 25).count(), 2);
    assert.deepEqual(
      await case13.locator('[data-slot="trace-description"]').allTextContents(),
      ['1.1s', '2.3s'],
      'results close the earliest running call first',
    );
    assert.equal(
      await case13.getByText('正在检索徒步知识库', { exact: true }).count(),
      0,
    );
    ok('parallel same-tool calls pair with their results in FIFO order');

    // 14. Thought chunks merge into one reasoning paragraph.
    const case14 = traceRoot(page, 27);
    const case14Trigger = case14.getByRole('button', { name: '思考过程', exact: true });
    assert.equal(await case14Trigger.count(), 1);
    assert.equal(await case14Trigger.getAttribute('aria-expanded'), 'false');
    await case14Trigger.click();
    await case14.getByText('思考', { exact: true }).waitFor();
    const detail = case14.locator('[data-slot="trace-detail"]');
    assert.equal(await detail.count(), 1);
    const reasoning = await detail.innerText();
    assert.match(reasoning, /先确认用户问的是雪线徒步/);
    assert.match(reasoning, /检索知识库中的雪线注意事项/);
    ok('thought events merge into one reasoning paragraph');

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

  // Dark mode with normal motion.
  const dark = await launch({ colorScheme: 'dark' });
  try {
    await seed(dark.page);
    await dark.page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    await dark.page.getByText('Case 10 answer', { exact: true }).waitFor();
    const darkTrace = traceRoot(dark.page, 1);
    assert.equal(await darkTrace.count(), 1);
    assert.equal(await darkTrace.getByText('正在思考', { exact: true }).count(), 1);
    assert.equal(
      await traceRoot(dark.page, 7).getByText('检索徒步知识库失败', { exact: true }).count(),
      1,
    );
    await dark.page.waitForTimeout(250);
    assert.deepEqual(dark.errors, [], 'no page or console errors in dark mode');
    ok('dark mode renders the trace with no errors');
  } finally {
    await dark.browser.close();
  }

  // Reduced motion: no shimmer and no collapse animation.
  const reduced = await launch({ colorScheme: 'light', reducedMotion: 'reduce' });
  try {
    await seed(reduced.page);
    await reduced.page.goto(BASE_URL, { waitUntil: 'domcontentloaded' });
    await reduced.page.getByText('Case 10 answer', { exact: true }).waitFor();
    const reducedTrace = traceRoot(reduced.page, 1);
    assert.equal(await reducedTrace.count(), 1);
    assert.equal(await reducedTrace.getByText('正在思考', { exact: true }).count(), 1);
    const reducedShimmer = await reducedTrace
      .locator('[data-slot="trace-title"]')
      .first()
      .evaluate(node => getComputedStyle(node).animationName);
    assert.equal(reducedShimmer, 'none', 'reduced motion disables the shimmer');
    const reducedCollapse = await traceRoot(reduced.page, 7)
      .locator('[data-slot="trace-collapsible-content"]')
      .evaluate(node => getComputedStyle(node).animationName);
    assert.equal(
      reducedCollapse,
      'none',
      'reduced motion disables the collapse animation',
    );
    await reduced.page.waitForTimeout(250);
    assert.deepEqual(reduced.errors, [], 'no page or console errors with reduced motion');
    ok('reduced motion disables the running-title shimmer');
    ok('reduced motion disables the collapse animation');
  } finally {
    await reduced.browser.close();
  }

  console.log(`\nAll ${passed} agent activity trace checks passed.`);
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
