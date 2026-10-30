// BCDEF ticket 11 — SSE recovery & client cancel/gap behavior.
// Run with: npm run build, then npx vite preview --port 4173, then
//   BASE_URL=http://127.0.0.1:4173 node tests/sse-recovery.cjs
// CHROME_PATH optionally selects an installed browser instead of Chromium.
//
// Scenarios (all route-mocked; zero real backend, zero real model calls):
//  1. live stream closes WITHOUT a terminal → snapshot + /events
//     reconciliation: no fake [连接错误], run not lost, no re-POST.
//  2. refresh over a dropped-progress hole → visible gap notice
//     (server-replayed progress.gap), single terminal, no extra SSE POST.
//  3. duplicate terminals in replay → rendered exactly once.
//  4. paused run refresh → approval prompt restored from snapshot
//     pending_approvals; persisted cursor NOT advanced past the pause.
//  5. outcome_unknown snapshot → first-class [结果未知] rendering, no
//     approval/resume affordance, no automatic model call.
//  6. snapshot done + terminal_persist_failed → done suppressed, stable
//     error shown, run marked non-complete (not streaming).
//  7. observation stream closes without a terminal (server-side a
//     losing/duplicate connection is read-only) → snapshot stays
//     RUNNING first, then converges to done — no fabricated recovery
//     failure, exactly ONE /chat/sse POST (no duplicate model submit).
//  8. approval decision → resume starts at the pause sequence (not the
//     held-back AC6 cursor), so a replayed run.paused never aborts the
//     continuation.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');

// 与 /api/auth/me fixture 的 test-user 保持一致：会话存储按账号命名空间隔离。
const RUN_KEY = 'u:test-user:ai-hiking-agent-v2-active-run';
const CHAT_KEY = 'u:test-user:ai-hiking-agent-chat';
const CHAT_ID_KEY = 'u:test-user:ai-hiking-agent-chat-id';

const env = (runId, seq, type, payload = {}) => ({
  version: '2',
  event_id: `${runId}-ev-${seq}`,
  sequence: seq,
  type,
  payload,
  request_id: 'req-fix',
  conversation_id: 'chat-fix',
  run_id: runId,
  timestamp: '2026-10-23T00:00:00+00:00',
});

const sseBody = events => events.map(e => `data: ${JSON.stringify(e)}\n\n`).join('');

const snapshotBase = (over = {}) => ({
  run_id: 'run-fix',
  request_id: 'req-fix',
  conversation_id: 'chat-fix',
  profile_id: 'chat',
  status: 'running',
  terminal_type: null,
  created_at: '2026-10-23T00:00:00+00:00',
  updated_at: '2026-10-23T00:00:05+00:00',
  terminal_at: null,
  sequence_cursor: 0,
  messages: [
    { role: 'user', content: '恢复测试问题', sequence: 0, run_id: 'run-fix' },
  ],
  pending_actions: [],
  terminal_persisted: false,
  terminal_persist_failed: false,
  error_code: null,
  recoverable: false,
  state_reconciled: false,
  pending_approvals: [],
  ...over,
});

(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.CHROME_PATH,
  });
  const base = process.env.BASE_URL || 'http://localhost:5173';
  const results = [];

  const openPage = async ({ run, snapshot, snapshotFor, eventsFor, resumeEventsFor, sse, extraRoutes = {}, viewport }) => {
    const page = await browser.newPage({ viewport: viewport || { width: 1280, height: 900 } });
    page.setDefaultTimeout(8000);
    const errors = [];
    const calls = { sse: 0, snapshot: 0, events: 0, cancel: 0 };
    const eventsQueries = [];
    const resumeQueries = [];
    const snapshotStatuses = [];
    page.on('pageerror', error => errors.push(error.stack || String(error)));
    await page.route('**/*', route => {
      // JWT 鉴权门 fixture：浏览器测试默认已登录
      if (new URL(route.request().url()).pathname === '/api/auth/me') return route.fulfill({ json: { user_id: 'test-user', login_id: 'tester', created_at: '2026-01-01T00:00:00Z' } });
      const url = new URL(route.request().url());
      const p = url.pathname;
      if (p === '/api/v2/chat/sse') {
        calls.sse += 1;
        return route.fulfill({ contentType: 'text/event-stream', body: sseBody(sse || []) });
      }
      if (p.startsWith('/api/v2/chat/resume/')) {
        const fromSeq = Number(url.searchParams.get('from_seq') || '0');
        resumeQueries.push(fromSeq);
        return route.fulfill({
          contentType: 'text/event-stream',
          body: sseBody(resumeEventsFor ? resumeEventsFor(fromSeq) : []),
        });
      }
      if (p.startsWith('/api/v2/chat/snapshot/')) {
        calls.snapshot += 1;
        const body = snapshotFor
          ? snapshotFor(calls.snapshot)
          : snapshot;
        snapshotStatuses.push(body && body.status);
        return route.fulfill({ json: body });
      }
      if (p.startsWith('/api/v2/chat/events/')) {
        calls.events += 1;
        const fromSeq = Number(url.searchParams.get('from_seq') || '0');
        eventsQueries.push(fromSeq);
        return route.fulfill({ json: { run_id: 'run-fix', from_seq: fromSeq, events: eventsFor(fromSeq) } });
      }
      if (p.startsWith('/api/v2/chat/cancel/')) {
        calls.cancel += 1;
        return route.fulfill({ status: 200, json: { accepted: true, reason: 'cancelled', run_id: 'run-fix' } });
      }
      if (extraRoutes[p]) {
        return route.fulfill(extraRoutes[p]);
      }
      return p.startsWith('/api/') ? route.fulfill({ json: { enabled: false } }) : route.continue();
    });
    await page.addInitScript(
      ({ runKey, chatKey, chatIdKey, runSeed }) => {
        navigator.geolocation.getCurrentPosition = (_s, failure) => failure?.({ code: 1, message: 'fixture' });
        localStorage.setItem(chatIdKey, 'chat-fix');
        localStorage.setItem(chatKey, JSON.stringify([]));
        if (runSeed) localStorage.setItem(runKey, JSON.stringify(runSeed));
      },
      { runKey: RUN_KEY, chatKey: CHAT_KEY, chatIdKey: CHAT_ID_KEY, runSeed: run || null },
    );
    await page.goto(base);
    return { page, errors, calls, eventsQueries, resumeQueries, snapshotStatuses };
  };

  const contentTexts = async page =>
    page.locator('[data-full-text]').evaluateAll(nodes => nodes.map(n => n.getAttribute('data-full-text')));

  const readRun = async page =>
    page.evaluate(key => localStorage.getItem(key), RUN_KEY);

  // ---- 1. live disconnect → snapshot reconciliation (no fake error) ----------
  {
    const { page, errors, calls } = await openPage({
      run: null,
      sse: [
        env('run-fix', 1, 'run_started'),
        env('run-fix', 2, 'progress', { message: '正在检索' }),
        env('run-fix', 3, 'text', { content: '已流出的半段回答' }),
        // transport closes here — NO terminal frame.
      ],
      snapshot: snapshotBase({
        status: 'done',
        terminal_type: 'done',
        terminal_persisted: true,
        terminal_event_type: 'done',
        sequence_cursor: 4,
        messages: [
          { role: 'user', content: '恢复测试问题', sequence: 0, run_id: 'run-fix' },
          { role: 'assistant', content: '已流出的半段回答', sequence: 1, run_id: 'run-fix' },
        ],
      }),
      eventsFor: fromSeq => (fromSeq < 4 ? [env('run-fix', 4, 'done')] : []),
    });
    await page.getByRole('textbox', { name: '输入徒步问题' }).fill('恢复测试问题');
    await page.getByRole('button', { name: '发送消息' }).click();
    await page.waitForFunction(
      key => localStorage.getItem(key) === null,
      RUN_KEY,
      { timeout: 6000 },
    );
    const texts = await contentTexts(page);
    const assistant = texts.filter(t => t.includes('已流出的半段回答'));
    assert.ok(assistant.length === 1, `scenario 1: run content preserved exactly once, got ${JSON.stringify(texts)}`);
    assert.ok(
      texts.every(t => !t.includes('[连接错误]') && !t.includes('stream_closed_without_terminal')),
      `scenario 1: no faked connection error, got ${JSON.stringify(texts)}`,
    );
    assert.equal(calls.sse, 1, 'scenario 1: reconciliation never re-POSTs /chat/sse (no extra model call)');
    assert.deepEqual(errors, [], 'scenario 1: uncaught browser errors');
    await page.close();
    results.push('1 live disconnect → snapshot/replay reconciliation, no [连接错误], run kept');
  }

  // ---- 2. refresh over a progress hole → visible gap notice ------------------
  {
    const { page, errors, calls } = await openPage({
      run: {
        run_id: 'run-fix', request_id: 'req-fix', conversation_id: 'chat-fix',
        sequence_cursor: 1, user_message: '恢复测试问题', v2_status: 'running',
      },
      sse: [],
      snapshot: snapshotBase({
        status: 'done', terminal_type: 'done', terminal_persisted: true, sequence_cursor: 6,
      }),
      eventsFor: fromSeq => (fromSeq < 6 ? [
        // server replay surfaces the hole between 1 and 5.
        env('run-fix', 2, 'progress.gap', { from_sequence: 1, to_sequence: 5, dropped: 3 }),
        env('run-fix', 5, 'text', { content: '恢复后的完整回答' }),
        env('run-fix', 6, 'done'),
      ] : []),
    });
    await page.locator('[data-full-text]').first().waitFor();
    const texts = await contentTexts(page);
    assert.ok(
      texts.some(t => t.includes('[注意] 部分进度更新丢失（序列 1 → 5，共 3 条）')),
      `scenario 2: gap notice visible, got ${JSON.stringify(texts)}`,
    );
    assert.ok(
      texts.some(t => t.includes('恢复后的完整回答') && t.includes('部分进度更新丢失')),
      'scenario 2: gap notice and recovered answer share the single assistant turn',
    );
    assert.equal(texts.filter(t => t.includes('恢复后的完整回答')).length, 1, 'scenario 2: one assistant message');
    assert.equal(calls.sse, 0, 'scenario 2: recovery made no model call');
    await page.waitForFunction(key => localStorage.getItem(key) === null, RUN_KEY);
    assert.deepEqual(errors, [], 'scenario 2: uncaught browser errors');
    await page.close();
    results.push('2 dropped progress → replay progress.gap rendered as a visible notice');
  }

  // ---- 3. duplicate terminals render once ------------------------------------
  {
    const { page, errors } = await openPage({
      run: {
        run_id: 'run-fix', request_id: 'req-fix', conversation_id: 'chat-fix',
        sequence_cursor: 0, user_message: '恢复测试问题', v2_status: 'running',
      },
      sse: [],
      snapshot: snapshotBase({ status: 'done', terminal_persisted: true, sequence_cursor: 4 }),
      eventsFor: fromSeq => (fromSeq < 4 ? [
        env('run-fix', 1, 'run_started'),
        env('run-fix', 2, 'text', { content: '只应出现一次的终态回答' }),
        env('run-fix', 3, 'done'),
        env('run-fix', 4, 'done', { note: 'duplicate' }), // late second terminal
      ] : []),
    });
    await page.locator('[data-full-text]').first().waitFor();
    await page.waitForFunction(key => localStorage.getItem(key) === null, RUN_KEY);
    const texts = await contentTexts(page);
    const hits = texts.filter(t => t.includes('只应出现一次的终态回答'));
    assert.equal(hits.length, 1, 'scenario 3: exactly one assistant message');
    assert.equal(hits[0], '只应出现一次的终态回答', 'scenario 3: duplicate terminal added nothing to the rendered turn');
    assert.deepEqual(errors, [], 'scenario 3: uncaught browser errors');
    await page.close();
    results.push('3 duplicate terminal from replay ignored (single render, counted in reducer state)');
  }

  // ---- 4. paused approval survives refresh -----------------------------------
  {
    const approval = {
      approval_id: 'apr-9', run_id: 'run-fix', tool: 'sandbox_bash',
      args: { command: 'ls' }, risk_level: 'high',
      impact: '在沙箱内执行 Bash 命令：ls',
      expires_at: '2027-01-01T00:00:00+00:00',
      message: '需要您的批准才能在沙箱内执行 Bash 命令。',
    };
    const { page, errors, eventsQueries } = await openPage({
      run: {
        run_id: 'run-fix', request_id: 'req-fix', conversation_id: 'chat-fix',
        sequence_cursor: 2, user_message: '要审批的问题', v2_status: 'paused',
        pending_approval: null,
      },
      sse: [],
      snapshot: snapshotBase({ status: 'paused', sequence_cursor: 3, pending_approvals: [approval] }),
      eventsFor: fromSeq => (fromSeq < 3 ? [env('run-fix', 3, 'run.paused', { approval_id: 'apr-9' })] : []),
    });
    await page.getByText('工具审批：sandbox_bash', { exact: false }).waitFor();
    await page.getByRole('button', { name: '批准' }).waitFor();
    assert.ok(eventsQueries.includes(2), 'scenario 4: replay started from the held cursor (2), not past the pause');
    const stored = JSON.parse(await readRun(page));
    assert.equal(stored.sequence_cursor, 2, 'scenario 4: paused run keeps the persisted cursor un-advanced (AC6)');
    assert.equal(stored.pending_approval.approval_id, 'apr-9', 'scenario 4: approval restored from snapshot pending_approvals');
    assert.equal(stored.v2_status, 'paused');
    assert.deepEqual(errors, [], 'scenario 4: uncaught browser errors');
    await page.close();
    results.push('4 paused approval restored on refresh from snapshot pending_approvals; cursor held');
  }

  // ---- 5. outcome_unknown first-class, non-continuable ------------------------
  {
    const { page, errors, calls } = await openPage({
      run: {
        run_id: 'run-fix', request_id: 'req-fix', conversation_id: 'chat-fix',
        sequence_cursor: 1, user_message: '恢复测试问题', v2_status: 'running',
      },
      sse: [],
      snapshot: snapshotBase({
        status: 'outcome_unknown',
        sequence_cursor: 2,
        messages: [
          { role: 'user', content: '恢复测试问题', sequence: 0, run_id: 'run-fix' },
          { role: 'assistant', content: '外部调用已发出，结果未知', sequence: 1, run_id: 'run-fix' },
        ],
      }),
      eventsFor: () => [], // no terminal event row — the snapshot finalizes the run
    });
    await page.locator('[data-full-text]').first().waitFor();
    const texts = await contentTexts(page);
    const ou = texts.filter(t => t.includes('[结果未知]'));
    assert.equal(ou.length, 1, `scenario 5: outcome_unknown rendered once, got ${JSON.stringify(texts)}`);
    assert.ok(ou[0].includes('外部调用已发出，结果未知'), 'scenario 5: snapshot messages backfill the turn');
    assert.ok(!texts.some(t => t.includes('[连接错误]')), 'scenario 5: not a fake network error');
    assert.equal(await page.getByRole('button', { name: '批准' }).count(), 0, 'scenario 5: no approval/resume affordance');
    assert.equal(calls.sse, 0, 'scenario 5: outcome_unknown never triggers a model call');
    await page.waitForFunction(key => localStorage.getItem(key) === null, RUN_KEY);
    assert.deepEqual(errors, [], 'scenario 5: uncaught browser errors');
    await page.close();
    results.push('5 outcome_unknown = first-class [结果未知] status, non-continuable, no model call');
  }

  // ---- 6. done suppressed when terminal_persist_failed ------------------------
  {
    const { page, errors } = await openPage({
      run: {
        run_id: 'run-fix', request_id: 'req-fix', conversation_id: 'chat-fix',
        sequence_cursor: 0, user_message: '恢复测试问题', v2_status: 'running',
      },
      sse: [],
      snapshot: snapshotBase({
        status: 'done',
        terminal_persisted: false,
        terminal_persist_failed: true,
        error_code: 'terminal_persist_failed',
        sequence_cursor: 2,
      }),
      eventsFor: fromSeq => (fromSeq < 2 ? [
        env('run-fix', 1, 'run_started'),
        env('run-fix', 2, 'text', { content: '内容有了但终态没落库' }),
      ] : []),
    });
    await page.locator('[data-full-text]').first().waitFor();
    const texts = await contentTexts(page);
    assert.ok(
      texts.some(t => t.includes('[错误] terminal_persist_failed')),
      `scenario 6: stable terminal_persist_failed error rendered, got ${JSON.stringify(texts)}`,
    );
    assert.ok(
      texts.every(t => !t.includes('已完成')),
      'scenario 6: done suppressed when the terminal never became durable (AC7)',
    );
    await page.waitForFunction(key => localStorage.getItem(key) === null, RUN_KEY);
    assert.deepEqual(errors, [], 'scenario 6: uncaught browser errors');
    await page.close();
    results.push('6 done suppressed on terminal_persist_failed; stable error, run non-complete');
  }

  // ---- 7. observation-stream close → running snapshot → done convergence --
  // agent-lifecycle-repair ticket 02 (T02-10): the server keeps exactly
  // one executor per run; a losing/duplicate connection is a READ-ONLY
  // observer whose stream replays persisted events and closes WITHOUT a
  // terminal while the real owner is still executing. The client must
  // converge via the existing close-without-terminal protocol: snapshot
  // still RUNNING → keep polling → DONE, with no fabricated recovery
  // failure and exactly ONE model submit (no duplicate POST /chat/sse).
  {
    const runningSnapshot = snapshotBase({
      status: 'running',
      sequence_cursor: 3,
      messages: [
        { role: 'user', content: '恢复测试问题', sequence: 0, run_id: 'run-fix' },
        { role: 'assistant', content: '观察期间已产出的回答', sequence: 1, run_id: 'run-fix' },
      ],
    });
    const doneSnapshot = snapshotBase({
      status: 'done',
      terminal_type: 'done',
      terminal_persisted: true,
      terminal_event_type: 'done',
      sequence_cursor: 4,
      messages: [
        { role: 'user', content: '恢复测试问题', sequence: 0, run_id: 'run-fix' },
        { role: 'assistant', content: '观察期间已产出的回答', sequence: 1, run_id: 'run-fix' },
      ],
    });
    // While snapshot #1 says running the owner is still executing: replay
    // has NOTHING new yet. The durable terminal event only exists once
    // the snapshot converged to done (served on call >= 2).
    const servedDone = { done: false };
    const { page, errors, calls, snapshotStatuses } = await openPage({
      run: null,
      // The observation stream: replayed persisted events, then the
      // server closes the read-only connection — no terminal frame.
      sse: [
        env('run-fix', 1, 'run_started'),
        env('run-fix', 2, 'progress', { message: '正在生成回复' }),
        env('run-fix', 3, 'text', { content: '观察期间已产出的回答' }),
      ],
      snapshotFor: call => {
        if (call === 1) return runningSnapshot;
        servedDone.done = true;
        return doneSnapshot;
      },
      eventsFor: fromSeq =>
        servedDone.done && fromSeq < 4 ? [env('run-fix', 4, 'done')] : [],
    });
    await page.getByRole('textbox', { name: '输入徒步问题' }).fill('恢复测试问题');
    await page.getByRole('button', { name: '发送消息' }).click();
    // Snapshot #1 says running → the client keeps polling (3s interval)
    // instead of fabricating an error or re-submitting.
    await page.waitForFunction(
      key => localStorage.getItem(key) === null,
      RUN_KEY,
      { timeout: 15000 },
    );
    const texts = await contentTexts(page);
    const answer = texts.filter(t => t.includes('观察期间已产出的回答'));
    assert.equal(answer.length, 1, `scenario 7: single recovered assistant turn, got ${JSON.stringify(texts)}`);
    assert.ok(
      texts.every(
        t => !t.includes('[连接错误]') && !t.includes('[恢复失败]') && !t.includes('stream_closed_without_terminal'),
      ),
      `scenario 7: no fabricated recovery failure, got ${JSON.stringify(texts)}`,
    );
    assert.equal(calls.sse, 1, 'scenario 7: exactly one model submit — the observation close never re-POSTs /chat/sse');
    assert.ok(
      snapshotStatuses[0] === 'running' && snapshotStatuses.includes('done'),
      `scenario 7: running → done snapshot convergence, got ${JSON.stringify(snapshotStatuses)}`,
    );
    assert.ok(calls.snapshot >= 2, 'scenario 7: the running snapshot was polled until done');
    assert.deepEqual(errors, [], 'scenario 7: uncaught browser errors');
    await page.close();
    results.push('7 observation-stream close → running snapshot polled to done; single model submit, no fake failure');
  }

  // ---- 8. approval resume starts at the pause sequence -----------------------
  {
    const approval = {
      approval_id: 'apr-8', run_id: 'run-fix', tool: 'mcp__custom__get_credits',
      args: {}, risk_level: 'high',
      impact: '执行 MCP 工具 mcp__custom__get_credits',
      expires_at: '2027-01-01T00:00:00+00:00',
      message: '需要批准后才能执行 MCP 工具。',
    };
    const { page, errors, resumeQueries } = await openPage({
      run: null,
      sse: [
        env('run-fix', 1, 'run_started'),
        env('run-fix', 2, 'approval.required', approval),
        env('run-fix', 3, 'run.paused', { approval_id: 'apr-8' }),
      ],
      extraRoutes: {
        '/api/v2/approvals/apr-8/decision': {
          json: {
            approval_id: 'apr-8', run_id: 'run-fix', tool: approval.tool,
            decision: 'approve', status: 'approved', idempotent: false,
            resume_url: '/api/v2/chat/resume/run-fix',
          },
        },
      },
      // Real server replay semantics: a resume from the held-back cursor
      // re-delivers run.paused and would abort the continuation.
      resumeEventsFor: fromSeq =>
        [
          ...(fromSeq < 3 ? [env('run-fix', 3, 'run.paused', { approval_id: 'apr-8' })] : []),
          env('run-fix', 4, 'approval.resolved', {
            approval_id: 'apr-8', decision: 'approve', status: 'approved',
          }),
          env('run-fix', 5, 'text', { content: '审批后继续的回答' }),
          env('run-fix', 6, 'done'),
        ],
    });
    await page.getByRole('textbox', { name: '输入徒步问题' }).fill('要审批的图片请求');
    await page.getByRole('button', { name: '发送消息' }).click();
    await page.getByRole('button', { name: '批准' }).waitFor();
    const held = JSON.parse(await readRun(page));
    assert.equal(held.sequence_cursor, 2, 'scenario 8: the paused run keeps the held-back cursor');
    await page.getByRole('button', { name: '批准' }).click();
    await page.locator('[data-full-text]').filter({ hasText: '审批后继续的回答' }).waitFor();
    assert.deepEqual(resumeQueries, [3], `scenario 8: resume must start at the pause sequence 3, got ${JSON.stringify(resumeQueries)}`);
    assert.deepEqual(errors, [], 'scenario 8: uncaught browser errors');
    await page.close();
    results.push('8 approval resume starts at the pause sequence (stale run.paused replay cannot abort it)');
  }

  // Expand a collapsed activity trace so its lines can be asserted.
  const openTrace = async page => {
    const trace = page.locator('[data-slot="agent-activity"]').first();
    await trace.waitFor();
    const trigger = trace.locator('[data-slot="trace-collapsible-trigger"]').first();
    if ((await trigger.getAttribute('aria-expanded')) === 'false') {
      await trigger.click();
    }
    return trace;
  };

  // ---- 9. mid-run refresh rebuilds the tool lines (ticket 05) ----------------
  {
    const runningSnapshot = snapshotBase({
      status: 'running',
      sequence_cursor: 3,
      messages: [
        { role: 'user', content: '恢复测试问题', sequence: 0, run_id: 'run-fix' },
      ],
    });
    const doneSnapshot = snapshotBase({
      status: 'done',
      terminal_type: 'done',
      terminal_persisted: true,
      terminal_event_type: 'done',
      sequence_cursor: 5,
      messages: [
        { role: 'user', content: '恢复测试问题', sequence: 0, run_id: 'run-fix' },
        { role: 'assistant', content: '恢复后的完整回答', sequence: 4, run_id: 'run-fix' },
      ],
    });
    const { page, errors, calls } = await openPage({
      run: {
        run_id: 'run-fix', request_id: 'req-fix', conversation_id: 'chat-fix',
        sequence_cursor: 1, user_message: '恢复测试问题', v2_status: 'running',
      },
      sse: [],
      snapshotFor: call => (call === 1 ? runningSnapshot : doneSnapshot),
      // The server replay is complete and ordered; the cursor decides where
      // it starts. The snapshot only decides the run's control status.
      eventsFor: fromSeq => [
        ['tool_call', { tool: 'hiking_knowledge_search', args: { query: '雪线' }, step: 0 }],
        ['tool_result', {
          tool: 'hiking_knowledge_search', status: 'ok', duration_ms: 1200,
          evidence: { has_chunks: true, chunks: [{}, {}] },
        }],
        ['text', { content: '恢复后的完整回答' }],
        ['done', {}],
      ]
        .map(([type, payload], index) => env('run-fix', index + 2, type, payload))
        .filter(event => event.sequence > fromSeq),
    });
    await page.locator('[data-full-text]').filter({ hasText: '恢复后的完整回答' }).waitFor();
    const trace = await openTrace(page);
    const lines = trace.locator('[data-slot="trace-collapsible-content"] [data-slot="trace-line"]');
    assert.equal(await lines.count(), 1, 'scenario 9: exactly one rebuilt tool line');
    assert.equal(
      await trace.getByText('已检索徒步知识库', { exact: true }).count(),
      1,
      'scenario 9: the finished call line survived the refresh',
    );
    assert.equal(
      await trace.getByText('查询「雪线」 · 1.2s · 2 条', { exact: true }).count(),
      1,
      'scenario 9: arguments and result detail were rebuilt together',
    );
    assert.equal(calls.sse, 0, 'scenario 9: recovery made no model call');
    await page.waitForFunction(key => localStorage.getItem(key) === null, RUN_KEY);
    assert.deepEqual(errors, [], 'scenario 9: uncaught browser errors');
    await page.close();
    results.push('9 mid-run refresh rebuilds the tool call + result line exactly once');
  }

  // ---- 10. model rounds, reasoning and the wait line survive refresh --------
  {
    const approval = {
      approval_id: 'apr-10', run_id: 'run-fix', tool: 'sandbox_bash',
      args: { command: 'ls' }, risk_level: 'high',
      impact: '在沙箱内执行 Bash 命令：ls',
      expires_at: '2027-01-01T00:00:00+00:00',
      message: '需要您的批准才能在沙箱内执行 Bash 命令。',
    };
    const { page, errors } = await openPage({
      run: {
        run_id: 'run-fix', request_id: 'req-fix', conversation_id: 'chat-fix',
        sequence_cursor: 1, user_message: '要审批的问题', v2_status: 'paused',
      },
      sse: [],
      snapshot: snapshotBase({ status: 'paused', sequence_cursor: 7, pending_approvals: [approval] }),
      eventsFor: fromSeq => (fromSeq < 7 ? [
        env('run-fix', 2, 'progress', {
          stage: 'model', step: 0, state: 'started', message: '模型第 1 轮',
        }),
        env('run-fix', 3, 'thought', { content: '先确认用户要执行什么命令。' }),
        env('run-fix', 4, 'progress', { stage: 'model', step: 0, state: 'finished', duration_ms: 800 }),
        env('run-fix', 5, 'tool_call', {
          tool: 'hiking_knowledge_search', args: { query: '雪线' }, step: 0,
        }),
        env('run-fix', 6, 'tool_result', {
          tool: 'hiking_knowledge_search', status: 'ok', duration_ms: 600,
          evidence: { has_chunks: true, chunks: [{}] },
        }),
        env('run-fix', 7, 'approval.required', approval),
      ] : []),
    });
    await page.getByRole('button', { name: '批准' }).waitFor();
    const trace = await openTrace(page);
    assert.equal(
      await trace.getByText('等待你的确认', { exact: true }).count(),
      1,
      'scenario 10: the approval wait summary is back',
    );
    assert.equal(await trace.getByText('模型第 1 轮', { exact: true }).count(), 1);
    assert.equal(
      await trace.getByText('800ms', { exact: true }).count(),
      1,
      'scenario 10: the model round kept its duration',
    );
    assert.equal(await trace.getByText('已检索徒步知识库', { exact: true }).count(), 1);
    assert.equal(await trace.getByText('思考', { exact: true }).count(), 1);
    assert.equal(
      await trace.getByText('等待确认：执行沙箱命令', { exact: true }).count(),
      1,
      'scenario 10: the pending wait line is preserved',
    );
    assert.equal(
      await trace.locator('[data-slot="trace-collapsible-content"] [data-slot="trace-line"]').count(),
      4,
      'scenario 10: model round + tool + thought + wait, no duplicate rows',
    );
    assert.deepEqual(errors, [], 'scenario 10: uncaught browser errors');
    await page.close();
    results.push('10 paused refresh restores model round, tool, reasoning and wait lines without duplicates');
  }

  // ---- 11. legacy run (results only) degrades without inventing rows --------
  {
    const { page, errors } = await openPage({
      run: {
        run_id: 'run-fix', request_id: 'req-fix', conversation_id: 'chat-fix',
        sequence_cursor: 0, user_message: '旧会话问题', v2_status: 'running',
      },
      sse: [],
      snapshot: snapshotBase({
        status: 'done',
        terminal_type: 'done',
        terminal_persisted: true,
        sequence_cursor: 3,
        messages: [
          { role: 'user', content: '旧会话问题', sequence: 0, run_id: 'run-fix' },
          { role: 'assistant', content: '旧会话回答', sequence: 2, run_id: 'run-fix' },
        ],
      }),
      // Legacy replay: a result row only — no tool_call, no rounds, no thoughts.
      eventsFor: fromSeq => (fromSeq < 3 ? [
        env('run-fix', 1, 'tool_result', {
          tool: 'hiking_knowledge_search', status: 'ok', duration_ms: 900,
          evidence: { has_chunks: true, chunks: [{}] },
        }),
        env('run-fix', 3, 'done'),
      ] : []),
    });
    await page.locator('[data-full-text]').filter({ hasText: '旧会话回答' }).waitFor();
    const trace = await openTrace(page);
    assert.equal(
      await trace.getByText('已检索徒步知识库', { exact: true }).count(),
      1,
      'scenario 11: the legacy result row still renders',
    );
    assert.equal(
      await trace.getByText('900ms · 1 条', { exact: true }).count(),
      1,
      'scenario 11: the legacy row keeps its duration and evidence count',
    );
    assert.equal(
      await trace.getByText('模型第 1 轮', { exact: false }).count(),
      0,
      'scenario 11: no model round is invented for a legacy run',
    );
    assert.equal(
      await trace.locator('[data-slot="trace-collapsible-content"] [data-slot="trace-line"]').count(),
      1,
      'scenario 11: exactly the existing subset',
    );
    assert.deepEqual(errors, [], 'scenario 11: uncaught browser errors');
    await page.close();
    results.push('11 legacy results-only replay degrades to that subset without invented rows');
  }

  // ---- 12. 390px: the rebuilt timeline does not overflow the page ----------
  {
    const approval = {
      approval_id: 'apr-12', run_id: 'run-fix', tool: 'sandbox_bash',
      args: { command: 'ls -la /tmp' }, risk_level: 'high',
      impact: '在沙箱内执行 Bash 命令：ls -la /tmp',
      expires_at: '2027-01-01T00:00:00+00:00',
      message: '需要您的批准才能在沙箱内执行 Bash 命令。',
    };
    const { page, errors } = await openPage({
      viewport: { width: 390, height: 844 },
      run: {
        run_id: 'run-fix', request_id: 'req-fix', conversation_id: 'chat-fix',
        sequence_cursor: 1, user_message: '要审批的问题', v2_status: 'paused',
      },
      sse: [],
      snapshot: snapshotBase({ status: 'paused', sequence_cursor: 7, pending_approvals: [approval] }),
      eventsFor: fromSeq => (fromSeq < 7 ? [
        env('run-fix', 2, 'progress', {
          stage: 'model', step: 0, state: 'started', message: '模型第 1 轮',
        }),
        env('run-fix', 3, 'progress', { stage: 'model', step: 0, state: 'finished', duration_ms: 700 }),
        env('run-fix', 5, 'tool_call', {
          tool: 'hiking_knowledge_search',
          args: { query: '雪线冬季露营装备清单与注意事项' },
          step: 0,
        }),
        env('run-fix', 6, 'tool_result', {
          tool: 'hiking_knowledge_search', status: 'ok', duration_ms: 600,
          evidence: { has_chunks: true, chunks: [{}] },
        }),
        env('run-fix', 7, 'approval.required', approval),
      ] : []),
    });
    await page.getByRole('button', { name: '批准' }).waitFor();
    const trace = await openTrace(page);
    assert.equal(
      await trace.getByText('模型第 1 轮', { exact: true }).count(),
      1,
      'scenario 12: the rebuilt model row is visible at 390px',
    );
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    assert.ok(overflow <= 0, `scenario 12: no horizontal overflow at 390px, got ${overflow}px`);
    assert.deepEqual(errors, [], 'scenario 12: uncaught browser errors');
    await page.close();
    results.push('12 390px recovery renders the rebuilt timeline without horizontal overflow');
  }

  await browser.close();
  console.log(`PASS: sse-recovery — ${results.length} scenarios.`);
  for (const r of results) console.log(`  - ${r}`);
})().catch(error => { console.error(error); process.exitCode = 1; });
