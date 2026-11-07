import { API } from './config'

export type SessionLifecycleEvent =
  | 'session.created'
  | 'session.restored'
  | 'session.deleted'
  | 'session.closed'
  | 'session.pruned'
  | 'session.snapshot.failed'

export type SessionLifecycleReason =
  | 'new_session'
  | 'switch'
  | 'window_exit'
  | 'tab_hidden'
  | 'mount'
  | 'quota_exceeded'
  | 'storage_unavailable'

export interface SessionLifecyclePayload {
  event: SessionLifecycleEvent
  reason?: SessionLifecycleReason
  session_id?: string
  conversation_id?: string
  snapshot_id?: string
  superseded_snapshot_id?: string
  outgoing_session_id?: string
  outgoing_snapshot_id?: string
  destroyed_count?: number
  snapshot_bytes?: number
}

export function reportSessionEvent(
  event: SessionLifecyclePayload,
  options: { exit?: boolean } = {},
): void {
  let body: string
  try {
    body = JSON.stringify(event)
  } catch { return }
  if (options.exit && typeof navigator !== 'undefined' && typeof navigator.sendBeacon === 'function') {
    try {
      if (navigator.sendBeacon(API.sessionLifecycle, new Blob([body], { type: 'application/json' }))) return
    } catch { /* fall back to fetch */ }
  }
  try {
    void fetch(API.sessionLifecycle, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
      credentials: 'same-origin',
      keepalive: true,
    }).catch(() => undefined)
  } catch { /* auditing must never break session actions */ }
}
