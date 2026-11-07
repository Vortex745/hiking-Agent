// Session snapshot lifecycle: restore, dedupe, prune, exit save and audit.
// Run with CHROME_PATH and BASE_URL pointing at a built or dev frontend.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');

const BASE = process.env.BASE_URL || 'http://localhost:5173';
const SESSIONS_KEY = 'u:test-user:ai-hiking-agent-sessions';

const userMessage = content => ({ role: 'user', content });

function session(id, title, content, overrides = {}) {
  return {
    id,
    chatId: `chat-${id}`,
    title,
    date: '2026-11-07',
    messages: [userMessage(content)],
    snapshotId: `snap-${id}-old`,
    savedAt: '2026-11-07T00:00:00Z',
    ...overrides,
  };
}

async function waitForAudit(audits, event, timeout = 5000) {
  return waitForAuditWhere(audits, item => item && item.event === event, event, timeout);
}

async function waitForAuditWhere(audits, predicate, label, timeout = 5000) {
  const started = Date.now();
  while (Date.now() - started < timeout) {
    const found = audits.find(predicate);
    if (found) return found;
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  throw new Error(`timeout waiting for ${label}; audits=${JSON.stringify(audits)}`);
}

async function readSessions(page) {
  return page.evaluate(key => {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : [];
  }, SESSIONS_KEY);
}

async function openPage(browser, {
  sessions,
  activeSessionId,
  chatId,
  messages,
  lifecycleStatus = 204,
  failSessionWrites = false,
  prefixed = false,
}) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.setDefaultTimeout(8000);
  const errors = [];
  const audits = [];
  page.on('pageerror', error => errors.push(error.stack || String(error)));
  page.on('request', request => {
    const pathname = new URL(request.url()).pathname;
    if (pathname !== '/api/v2/sessions/lifecycle') return;
    try {
      audits.push(request.postDataJSON());
    } catch {
      audits.push(null);
    }
  });
  await page.route('**/*', route => {
    if (new URL(route.request().url()).hostname === 'fonts.googleapis.com') {
      return route.abort();
    }
    const pathname = new URL(route.request().url()).pathname;
    if (pathname === '/api/auth/me') {
      return route.fulfill({
        json: {
          user_id: 'test-user',
          login_id: 'tester',
          created_at: '2026-01-01T00:00:00Z',
        },
      });
    }
    if (pathname === '/api/v2/sessions/lifecycle') {
      return lifecycleStatus === 204
        ? route.fulfill({ status: 204 })
        : route.fulfill({ status: lifecycleStatus, json: { error: 'fixture' } });
    }
    if (pathname === '/api/v2/memory/settings') {
      return route.fulfill({ json: { subject_id: 'test-user', enabled: false } });
    }
    return pathname.startsWith('/api/')
      ? route.fulfill({ json: { enabled: false } })
      : route.continue();
  });
  await page.addInitScript(
    ({ sessions, activeSessionId, chatId, messages, failSessionWrites, prefixed }) => {
      navigator.geolocation.getCurrentPosition = (_success, failure) =>
        failure?.({ code: 1, message: 'fixture' });
      const prefix = prefixed ? 'u:test-user:' : '';
      if (prefixed) localStorage.setItem('ai-hiking-storage-claimed', 'test-user');
      localStorage.setItem(`${prefix}ai-hiking-agent-sessions`, JSON.stringify(sessions));
      localStorage.setItem(`${prefix}ai-hiking-agent-active-session`, activeSessionId);
      localStorage.setItem(`${prefix}ai-hiking-agent-chat-id`, chatId);
      localStorage.setItem(`${prefix}ai-hiking-agent-chat`, JSON.stringify(messages));
      if (failSessionWrites) {
        const original = Storage.prototype.setItem;
        Storage.prototype.setItem = function (key, value) {
          if (String(key).includes('ai-hiking-agent-sessions')) {
            throw new DOMException('quota', 'QuotaExceededError');
          }
          return original.call(this, key, value);
        };
      }
    },
    { sessions, activeSessionId, chatId, messages, failSessionWrites, prefixed },
  );
  await page.goto(BASE);
  await page.getByRole('textbox', { name: '输入徒步问题' }).waitFor();
  return { page, audits, errors };
}

(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.CHROME_PATH,
  });
  try {
    {
      const sessions = Array.from({ length: 51 }, (_, index) =>
        session(`s${index}`, `Session ${index}`, `hello ${index}`, {
          savedAt: `2026-11-07T00:00:${String(index).padStart(2, '0')}Z`,
        }),
      );
      const duplicateOld = session('dup', 'Dup', 'old dup', {
        snapshotId: 'snap-dup-old',
        savedAt: '2026-11-07T00:00:00Z',
      });
      const duplicateNew = session('dup', 'Dup', 'new dup', {
        snapshotId: 'snap-dup-new',
        savedAt: '2026-11-07T00:01:00Z',
      });
      const seeded = [
        null,
        sessions[0],
        duplicateOld,
        sessions[1],
        duplicateNew,
        ...sessions.slice(2),
      ];
      const { page, audits, errors } = await openPage(browser, {
        sessions: seeded,
        activeSessionId: 's0',
        chatId: 'chat-s0',
        messages: sessions[0].messages,
      });
      await page.waitForFunction(
        key => {
          const raw = localStorage.getItem(key);
          if (!raw) return false;
          return JSON.parse(raw).length <= 50;
        },
        SESSIONS_KEY,
      );
      const stored = await readSessions(page);
      assert.equal(stored.length, 50, 'session index is capped at 50');
      assert.equal(new Set(stored.map(item => item.id)).size, 50, 'no duplicate session ids');
      const pruned = await waitForAudit(audits, 'session.pruned');
      assert.ok(pruned.destroyed_count >= 2, 'prune count reports destroyed entries');

      await page
        .getByRole('complementary', { name: '对话历史' })
        .getByText('Dup', { exact: true })
        .click();
      await page.locator('[data-full-text="new dup"]').waitFor();
      const restored = await waitForAudit(audits, 'session.restored');
      assert.equal(restored.session_id, 'dup');
      assert.equal(restored.snapshot_id, 'snap-dup-new');

      await page
        .getByRole('complementary', { name: '对话历史' })
        .getByText('Session 1', { exact: true })
        .click();
      await page.locator('[data-full-text="hello 1"]').waitFor();
      const afterSwitch = await readSessions(page);
      const previous = afterSwitch.find(item => item.id === 'dup');
      assert.ok(previous.snapshotId && previous.snapshotId !== 'snap-dup-new');
      assert.ok(previous.savedAt);
      assert.deepEqual(errors, []);
      await page.close();
      console.log('1 restore/dedupe/prune passed');
    }

    {
      const alpha = session('alpha', 'Alpha', 'alpha latest');
      const { page, audits, errors } = await openPage(browser, {
        sessions: [alpha],
        activeSessionId: 'alpha',
        chatId: 'chat-alpha',
        messages: alpha.messages,
      });
      await page.getByRole('button', { name: '新建对话' }).click();
      const created = await waitForAudit(audits, 'session.created');
      assert.equal(created.outgoing_session_id, 'alpha');
      assert.equal(created.superseded_snapshot_id, 'snap-alpha-old');
      assert.equal(created.snapshot_id, undefined);
      const stored = await readSessions(page);
      const saved = stored.find(item => item.id === 'alpha');
      assert.ok(saved.snapshotId && saved.snapshotId !== 'snap-alpha-old');
      assert.equal(created.outgoing_snapshot_id, saved.snapshotId);
      assert.equal(saved.messages.at(-1).content, 'alpha latest');
      await page
        .getByRole('complementary', { name: '对话历史' })
        .getByText('Alpha', { exact: true })
        .click();
      await page.locator('[data-full-text="alpha latest"]').waitFor();
      await waitForAudit(audits, 'session.restored');
      assert.deepEqual(errors, []);
      await page.close();
      console.log('2 new-session snapshot passed');
    }

    {
      const alpha = session('alpha', 'Alpha', 'alpha answer');
      const beta = session('beta', 'Beta', 'beta answer', {
        messages: [
          userMessage('beta question'),
          { role: 'assistant', content: 'beta answer', isStreaming: true },
        ],
      });
      const { page, audits, errors } = await openPage(browser, {
        sessions: [alpha, beta],
        activeSessionId: 'alpha',
        chatId: 'chat-alpha',
        messages: alpha.messages,
      });
      await page
        .getByRole('complementary', { name: '对话历史' })
        .getByText('Beta', { exact: true })
        .click();
      await page.locator('[data-full-text="beta answer"]').waitFor();
      await page.evaluate(() => window.dispatchEvent(new PageTransitionEvent('pagehide')));
      const closed = await waitForAudit(audits, 'session.closed');
      assert.equal(closed.session_id, 'beta');
      const stored = await readSessions(page);
      const savedBeta = stored.find(item => item.id === 'beta');
      assert.ok(savedBeta.snapshotId && savedBeta.snapshotId !== 'snap-beta-old');
      assert.ok(savedBeta.messages.every(message => message.isStreaming !== true));
      await page.evaluate(() => window.dispatchEvent(new PageTransitionEvent('pagehide')));
      await page.waitForTimeout(300);
      assert.equal(
        audits.filter(item => item && item.event === 'session.closed').length,
        1,
        'one close event per unload',
      );
      assert.deepEqual(errors, []);
      await page.close();
      console.log('3 pagehide snapshot passed');
    }

    {
      const alpha = session('alpha', 'Alpha', 'alpha answer');
      const beta = session('beta', 'Beta', 'beta answer');
      const { page, audits, errors } = await openPage(browser, {
        sessions: [alpha, beta],
        activeSessionId: 'alpha',
        chatId: 'chat-alpha',
        messages: alpha.messages,
      });
      const sidebar = page.getByRole('complementary', { name: '对话历史' });
      await sidebar
        .getByText('Beta', { exact: true })
        .locator('..')
        .locator('..')
        .getByRole('button', { name: '删除对话' })
        .click();
      const deleted = await waitForAudit(audits, 'session.deleted');
      assert.equal(deleted.session_id, 'beta');
      assert.equal((await readSessions(page)).some(item => item.id === 'beta'), false);
      await sidebar
        .getByText('Alpha', { exact: true })
        .locator('..')
        .locator('..')
        .getByRole('button', { name: '删除对话' })
        .click();
      await page.getByRole('textbox', { name: '输入徒步问题' }).waitFor();
      await page.waitForTimeout(100);
      assert.equal(
        (await readSessions(page)).some(item => item.id === 'alpha'),
        false,
        'active session snapshot is deleted',
      );
      assert.deepEqual(errors, []);
      await page.close();
      console.log('4 delete snapshot passed');
    }

    {
      const alpha = session('alpha', 'Alpha', 'alpha answer');
      const beta = session('beta', 'Beta', 'beta answer');
      const { page, errors } = await openPage(browser, {
        sessions: [alpha, beta],
        activeSessionId: 'alpha',
        chatId: 'chat-alpha',
        messages: alpha.messages,
        lifecycleStatus: 500,
      });
      await page
        .getByRole('complementary', { name: '对话历史' })
        .getByText('Beta', { exact: true })
        .click();
      await page.locator('[data-full-text="beta answer"]').waitFor();
      assert.deepEqual(errors, []);
      await page.close();
      console.log('5 audit failure isolation passed');
    }

    {
      const alpha = session('alpha', 'Alpha', 'alpha answer');
      const beta = session('beta', 'Beta', 'beta answer');
      const { page, audits, errors } = await openPage(browser, {
        sessions: [alpha, beta],
        activeSessionId: 'alpha',
        chatId: 'chat-alpha',
        messages: alpha.messages,
        failSessionWrites: true,
        prefixed: true,
      });
      const betaBytes = await page.evaluate(({ key, id }) => {
        const list = JSON.parse(localStorage.getItem(key) || '[]');
        const entry = list.find(item => item.id === id);
        return new TextEncoder().encode(JSON.stringify(entry)).byteLength;
      }, { key: SESSIONS_KEY, id: 'beta' });
      await page
        .getByRole('complementary', { name: '对话历史' })
        .getByText('Beta', { exact: true })
        .click();
      await page.locator('[data-full-text="beta answer"]').waitFor();
      const failed = await waitForAuditWhere(
        audits,
        item => item
          && item.event === 'session.snapshot.failed'
          && item.session_id === 'alpha',
        'session.snapshot.failed for the outgoing session',
      );
      const restored = await waitForAudit(audits, 'session.restored');
      assert.equal(failed.reason, 'quota_exceeded');
      assert.equal(failed.conversation_id, 'chat-alpha');
      assert.equal(restored.session_id, 'beta');
      assert.equal(restored.snapshot_id, 'snap-beta-old');
      assert.equal(restored.snapshot_bytes, betaBytes);
      assert.deepEqual(errors, []);
      await page.close();
      console.log('6 storage failure isolation passed');
    }

    console.log('PASS: session snapshot lifecycle');
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
