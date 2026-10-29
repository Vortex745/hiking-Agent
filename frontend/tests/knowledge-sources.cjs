// Browser regression for the local-only knowledge-source manager.
// Run with: CHROME_PATH=<chrome> BASE_URL=http://localhost:5175 node tests/knowledge-sources.cjs
const { chromium } = require('playwright')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const errors = []
const counts = {
  discovered: 1,
  fetched: 1,
  created: 1,
  updated: 0,
  skipped: 0,
  skipped_by_type: 0,
  truncated: 0,
  failed: 0,
  deleted: 0,
}
const runningLog = {
  id: 'sync-1',
  datasource_id: 'ds-1',
  trigger: 'manual',
  status: 'running',
  started_at: '2026-10-30T10:00:00+08:00',
  finished_at: null,
  counts: { ...counts, created: 0, fetched: 0, discovered: 0 },
  failures: [],
  enumeration: {},
  error_code: null,
  message: null,
}
const successLog = {
  ...runningLog,
  status: 'success',
  finished_at: '2026-10-30T10:00:05+08:00',
  counts,
}

const types = [
  {
    type: 'fake',
    label: '测试来源',
    auth_fields: [
      { key: 'token', label: 'Token', secret: true, required: true, placeholder: '', help: '' },
    ],
    setting_fields: [],
    supports_tree: true,
    supports_incremental: true,
    supports_deletion: true,
  },
]
const roots = [
  { id: 'root', title: '测试空间', kind: 'space', parent_id: null, has_children: true, rev: '', url: '' },
]
const children = [
  { id: 'doc-1', title: '路线文档', kind: 'document', parent_id: 'root', has_children: false, rev: '1', url: '' },
]
const baseSource = {
  id: 'ds-1',
  type: 'fake',
  name: '测试语雀',
  credentials_configured: { token: true },
  settings: {},
  selected_resources: ['root'],
  schedule: '',
  sync_deletions: true,
  sync_status: 'idle',
  running_since: null,
  next_run_at: null,
  last_sync_at: null,
  last_sync_status: null,
  last_sync: null,
  doc_count: 0,
  created_at: '2026-10-30T10:00:00+08:00',
  updated_at: '2026-10-30T10:00:00+08:00',
}

let source = null
let syncCalls = 0

const json = (route, body, status = 200) =>
  route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })

;(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.CHROME_PATH,
  })
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
    page.setDefaultTimeout(15000)
    page.on('pageerror', (error) => errors.push(error.stack || String(error)))
    page.on('console', (message) => {
      if (message.type() === 'error') errors.push(`console: ${message.text()}`)
    })

    await page.route('**/*', (route) => {
      const request = route.request()
      const url = new URL(request.url())
      const pathname = url.pathname
      const method = request.method()
      if (!pathname.startsWith('/api/')) return route.continue()

      if (pathname === '/api/v1/datasources/types') return json(route, types)
      if (pathname === '/api/v1/datasources/preview/validate') {
        return json(route, { ok: true, message: '连接成功' })
      }
      if (pathname === '/api/v1/datasources/preview/resources') {
        const body = request.postDataJSON()
        return json(route, body.parent_id ? children : roots)
      }
      if (pathname === '/api/v1/datasources/preview/ancestors') return json(route, [])
      if (pathname === '/api/v1/datasources' && method === 'GET') {
        return json(route, source ? [source] : [])
      }
      if (pathname === '/api/v1/datasources' && method === 'POST') {
        const body = request.postDataJSON()
        source = {
          ...baseSource,
          name: body.name,
          selected_resources: body.selected_resources,
          schedule: body.schedule,
          sync_deletions: body.sync_deletions,
        }
        return json(route, source, 201)
      }
      if (pathname.endsWith('/syncs/sync-1')) {
        syncCalls += 1
        if (syncCalls === 1) return json(route, runningLog)
        source = {
          ...source,
          sync_status: 'idle',
          last_sync_status: 'success',
          last_sync: successLog,
          last_sync_at: successLog.finished_at,
          doc_count: 1,
        }
        return json(route, successLog)
      }
      if (pathname.endsWith('/syncs')) return json(route, [successLog])
      if (pathname.endsWith('/sync') && method === 'POST') {
        source = { ...source, sync_status: 'running' }
        syncCalls = 0
        return json(route, { sync_id: 'sync-1', status: 'running' }, 202)
      }
      if (pathname.endsWith('/documents')) {
        return json(route, [
          {
            external_id: 'doc-1',
            resource_id: 'root',
            rev: '1',
            rag_source: 'ds:ds-1:doc-1',
            file_name: '路线文档.md',
            last_seen_sync_id: 'sync-1',
            updated_at: '2026-10-30T10:00:05+08:00',
          },
        ])
      }
      if (pathname.includes('/ancestors')) return json(route, [])
      if (pathname.includes('/resources')) return json(route, roots)
      if (method === 'PATCH') {
        const body = request.postDataJSON()
        source = { ...source, ...body }
        return json(route, source)
      }
      if (method === 'DELETE') return json(route, { ok: true, message: '已删除' })
      return json(route, { enabled: false })
    })

    await page.goto(process.env.BASE_URL || 'http://localhost:5175')
    await page.getByRole('button', { name: '知识源' }).click()
    await page.getByRole('dialog', { name: '知识源' }).waitFor()
    await page.getByRole('button', { name: '新建知识源' }).first().click()
    await page.getByPlaceholder('例如：徒步路线资料').fill('测试语雀')
    await page.locator('input[type="password"]').first().fill('tok-123')
    await page.getByRole('button', { name: '测试连接' }).click()
    await page.getByText('连接成功').waitFor()
    await page.getByRole('button', { name: '加载资源' }).click()
    await page.getByText('测试空间', { exact: true }).waitFor()
    await page.getByLabel('选择 测试空间').check()
    await page.getByRole('button', { name: '创建知识源' }).click()
    await page.getByText('测试语雀', { exact: true }).waitFor()

    await page.getByRole('button', { name: '同步' }).click()
    await page.getByText(/同步完成/).waitFor()
    await page.getByRole('button', { name: '日志' }).click()
    await page.getByText('手动同步', { exact: true }).waitFor()
    await page.getByText(/新增 1/).first().waitFor()

    await page.getByRole('button', { name: '返回列表' }).click()
    await page.getByRole('button', { name: '删除知识源 测试语雀' }).click()
    await page.getByText('删除「测试语雀」？').waitFor()
    await page.getByRole('button', { name: '取消' }).click()
    await page.getByRole('dialog', { name: '知识源' }).waitFor()

    const screenshotPath =
      process.env.SCREENSHOT_PATH || path.resolve(__dirname, '../../.screenshots/knowledge-sources.png')
    fs.mkdirSync(path.dirname(screenshotPath), { recursive: true })
    await page.screenshot({ path: screenshotPath })
    assert.deepEqual(errors, [], 'uncaught browser errors')
    console.log('PASS: knowledge source create, credential test, tree pick, sync polling, logs and delete confirmation.')
    console.log(`SCREENSHOT: ${screenshotPath}`)
  } finally {
    await browser.close()
  }
})().catch((error) => {
  console.error(error)
  console.error(errors.join('\n'))
  process.exitCode = 1
})
