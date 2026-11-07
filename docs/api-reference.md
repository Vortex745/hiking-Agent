# ai-service 接口文档

> 对应代码：`ai-service/main.py`、`ai-service/api/*.py`、`agent_v2/events.py`、`agent_v2/run_orchestration.py`（2026-10-22）。
> 仅描述当前实际注册的路由；v1 chat 已删除，Chat v2 是唯一对话链路。

## 1. 服务地址与通用约定

| 入口 | 地址 | 说明 |
| --- | --- | --- |
| 直连 ai-service | `http://localhost:8000` | `uvicorn` 固定 8000，端口写死在 `main.py` |
| Gateway | `http://localhost:8080` | Go/Gin，转发业务路径到 ai-service；`PORT` / `AI_SERVICE_URL` 可配。`/api/v2/chat/health` 与 `/api/v1/llm-config*` 未在 Gateway 注册，仅直连可用 |
| 前端 dev | `http://localhost:5173` | Vite 将 `/api` 代理到 `:8000` |

- OpenAPI：`GET /docs`、`GET /openapi.json`（FastAPI 默认开启）。
- 身份：httpOnly Cookie `hiking_subject`。首次访问任一 v2 接口时自动签发；同一 subject 才能访问自己的 run / conversation / memory / artifact。跨域调用需 `credentials: 'include'`。
- 普通 HTTP 错误体：`{"detail": "..."}`（FastAPI 默认）。
- SSE 内的业务错误不是 HTTP 错误：HTTP 200 开始流，错误通过终端 `error` 事件返回。
- 时间均为 UTC ISO 8601 字符串。

## 2. 接口总览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 服务与 v2 运行时健康状态 |
| POST | `/api/v2/chat/sse` | 发起一轮对话（SSE） |
| GET | `/api/v2/chat/health` | chat 路由轻量探活 |
| GET | `/api/v2/chat/snapshot/{run_id}` | 运行快照（消息、状态、游标） |
| GET | `/api/v2/chat/events/{run_id}` | 断线重连事件回放 |
| POST | `/api/v2/chat/cancel/{run_id}` | 幂等取消运行 |
| POST | `/api/v2/chat/resume/{run_id}` | 审批后恢复运行（SSE） |
| POST | `/api/v2/approvals/{approval_id}/decision` | 提交 approve / reject |
| GET | `/api/v2/artifacts/{artifact_id}` | 下载已发布 artifact |
| GET / PUT | `/api/v2/memory/settings` | 读取 / 修改记忆开关 |
| GET | `/api/v2/memories` | 列出记忆 |
| PATCH / DELETE | `/api/v2/memories/{memory_id}` | 修改 / 删除单条记忆 |
| DELETE | `/api/v2/memories?confirm=true` | 清空全部记忆 |
| GET | `/api/v2/memories/export` | 导出记忆为 JSON artifact |
| POST | `/api/v2/memory/candidates/{candidate_id}/decision` | 敏感记忆候选的同意 / 拒绝 |
| POST | `/api/v2/subject/rotate` | 轮换匿名身份 token |
| POST | `/api/v2/sessions/lifecycle` | 会话快照生命周期审计（仅日志） |
| POST | `/api/v1/models/fetch` | 用给定 base_url + key 拉取上游模型列表 |
| GET | `/api/v1/llm-config` | 读取 LLM / embedding / rerank 配置（密钥打码） |
| PUT | `/api/v1/llm-config` | 写入 `.env` 并热更新运行时配置 |
| POST | `/api/v1/llm-config/test` | 探测某段配置的上游 `/models` 连通性 |

## 3. SSE 通用契约

### 3.1 传输

- 请求：`POST`，`Content-Type: application/json`
- 响应：`200`，`Content-Type: text/event-stream`
- 响应头：`Cache-Control: no-cache`、`X-Accel-Buffering: no`
- 每帧格式：`data: <JSON>\n\n`
- 心跳：内核生成阶段每 15 秒发一次 `heartbeat`（前端据此保活）
- 正常运行结束时服务端关闭连接

### 3.2 事件信封

```json
{
  "version": "2",
  "event_id": "e1c54421-fed3-44a7-a759-c9f016f79d7b",
  "sequence": 1,
  "run_id": "1dcfcd92-2e0e-421b-8bc1-f0539b2164d2",
  "conversation_id": "",
  "timestamp": "2026-10-21T15:42:53.540900+00:00",
  "type": "run_started",
  "request_id": "client-key-1",
  "payload": {}
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `version` | string | 固定 `"2"` |
| `event_id` | string | 每条事件的服务端 UUID |
| `sequence` | int | 单次 run 内从 1 开始严格递增 |
| `run_id` | string | 服务端 run id；首个事件起携带 |
| `conversation_id` | string | 会话 id；未传时为空（内部派生） |
| `timestamp` | string | UTC ISO 8601 |
| `type` | string | 见下表 |
| `request_id` | string | 客户端幂等键 |
| `payload` | object | 事件负载，见下表 |

### 3.3 事件类型与 payload

| type | 何时出现 | payload |
| --- | --- | --- |
| `run_started` | 流开始 | `{}` |
| `progress` | 阶段提示 | `{stage, message}`，如 `{"stage":"processing","message":"正在生成回复..."}` |
| `heartbeat` | 模型思考 / 工具执行期间 | `{}` |
| `text` | 正文增量或完整文本 | `{content}`。内核流式为增量；artifact 流程为整段 |
| `tool_call` | 工具开始执行 | `{tool, args}` |
| `tool_result` | 工具完成 / 审计 | `{tool, status, content, ...}`，详见 3.4 |
| `artifact_candidate` | 沙箱内产物生成，等待导出 | `{filename, mime_type, sandbox_type, message}` |
| `artifact.available` | 宿主校验并持久化完成 | `{artifact_id, public_filename, mime_type, size_bytes, sha256, expires_at, message}` |
| `approval.required` | 高风险工具需人工批准 | `{approval_id, tool, args, risk_level, impact, expires_at, message}` |
| `approval.resolved` | 审批结果生效（resume 流） | `{approval_id, tool, decision, status, message}` |
| `memory.consent_required` | 检测到敏感信息 | `{candidate_id, category, preview, sensitivity, status, created_at, expires_at}` |
| `run.paused` | 等待审批（流结束，run 未结束） | `{code:"awaiting_approval", approval_id, message}`，终端事件 |
| `effect.outcome_unknown` | 副作用结果不确定 | `{effect_key, tool, message}`，其后仍会跟 `error` 终端 |
| `error` | 失败 | `{code, message, unknown_dropped}`，终端事件 |
| `done` | 成功结束 | `{unknown_dropped}`；拒绝审批时为 `{code:"rejected", message, unknown_dropped}` |
| `cancelled` | 取消 | `{code:"cancelled", message, unknown_dropped}`，终端事件 |

### 3.4 `tool_result` 常见形态

| 场景 | 字段 |
| --- | --- |
| 内核工具审计 | `{tool, status, content, duration_ms?, reason?, server_id?, evidence?}`；`status` 为 `ok`、`error`、`denied`、`mcp_server_unavailable` 或具体错误码 |
| `sandbox_bash` | `{tool:"sandbox_bash", status:"ok", exit_code, stdout, stderr, duration_ms, truncated, content}` |
| `generate_pdf` | `{tool:"generate_pdf", status:"ok", content, path}` |

`evidence`（检索来源）形如 `{has_chunks, sources, chunks}`，仅在有检索证据时出现。

### 3.5 终端规则

- 每个流**恰好**一个流终端：`error` / `done` / `cancelled` / `run.paused`。
- `run.paused` 结束当前流，但 run 生命周期未结束，需走审批 + resume。
- `effect.outcome_unknown` 不是终端，随后必有 `error`。
- 连接 EOF 但没看到终端事件时，客户端应视为 `outcome_unknown`，用 snapshot 对账。
- `error` 后不会再有 `done`/`text`；取消后不会泄漏迟到的正文。

## 4. 接口详情

### 4.1 `GET /health`

返回服务状态与 v2 运行时信息。健康 ≠ 可对话：模型 key 失效时它仍返回 200/ready。

```json
{
  "status": "ok",
  "service": "ai-service",
  "agent_v2": {
    "enabled": true,
    "status": "ready",
    "uniharness_pin": "b026a262...",
    "kernel_identity": {"distribution_version": "...", "fork_commit": "...", "compatible": true},
    "writer_id": "8b692ea9-...",
    "state_dir": "./agent_state",
    "persistence": {
      "runs_repository": true,
      "writer_lock_held": true,
      "writer_lock_row": {"lock_id": 1, "writer_id": "8b692ea9-...", "acquired_at": "...", "expires_at": null}
    },
    "dependencies": {
      "model": {"status": "ready"},
      "mcp": {"status": "disabled"},
      "sandbox": {"status": "trusted_local", "type": "local_native", "verified_isolation": false},
      "checkpoint": {"status": "ready"},
      "control": {"status": "ready"},
      "memory": {"status": "ready"},
      "artifact_store": {"status": "ready"}
    }
  }
}
```

`agent_v2.status`：`ready` / `failed` / `disabled`。`failed` 时不要路由 v2 流量。

### 4.2 `POST /api/v2/chat/sse`

发起一轮 Chat v2 运行。

**请求体**（`extra="forbid"`，多余字段直接 422）：

```json
{
  "message": "你好，请用一句话介绍你自己",
  "request_id": "client-key-1",
  "conversation_id": "conv-optional",
  "profile_id": "chat",
  "current_location": {
    "latitude": 30.25,
    "longitude": 120.16,
    "accuracy": 20,
    "province": "浙江省",
    "city": "杭州市",
    "district": "西湖区",
    "adcode": "330106",
    "address": "…",
    "source": "browser"
  }
}
```

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `message` | 是 | 非空用户消息 |
| `request_id` | 是 | 幂等键。重复 POST 会回放已存流，不会重复执行工具 |
| `conversation_id` | 否 | 多轮会话 id；必须属于当前 subject |
| `profile_id` | 否 | 默认 `chat`，见第 6 节；客户端只能“请求”，不能越权 |
| `current_location` | 否 | 所有子字段可选 |

**响应**：`text/event-stream`（见第 3 节）。

**HTTP 状态**：

| 状态 | 场景 | 响应体 |
| --- | --- | --- |
| 200 | 流已接受 | SSE |
| 422 | 请求体非法（缺字段、多余字段、类型错误） | `{"detail":[...]}` |
| 503 | 运行/沙箱容量耗尽 | `{"error":{"code":"run_capacity_exceeded" 或 "sandbox_capacity_exceeded","retryable":true}}` |

业务拒绝（`v2_disabled`、`unknown_profile`、`cowork_unauthorized`、`model_unavailable`、`conversation_forbidden` 等）都在 SSE 内以 `error` 终端返回。

**示例**：

```bash
curl -N -c jar -b jar \
  -H 'Content-Type: application/json; charset=utf-8' \
  -X POST http://127.0.0.1:8000/api/v2/chat/sse \
  --data-binary @chat.json
```

### 4.3 `GET /api/v2/chat/health`

轻量探活，不装配运行时、不检查模型 key。

```json
{"status": "ok", "module": "chat_v2", "service": "ai-service", "uniharness_pin": "..."}
```

v2 关闭时 `status` 为 `"disabled"`。

### 4.4 `GET /api/v2/chat/snapshot/{run_id}`

返回运行快照。路径参数也接受 `request_id`（用于客户端还没拿到 run_id 时）。

**Query**

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `request_id` | `""` | 路径 id 解析失败时的兜底幂等键 |

**响应 200**：

```json
{
  "run_id": "...",
  "request_id": "...",
  "conversation_id": "...",
  "profile_id": "chat",
  "status": "running",
  "terminal_type": null,
  "created_at": "...",
  "updated_at": "...",
  "terminal_at": null,
  "sequence_cursor": 12,
  "messages": [
    {"role": "user", "content": "...", "sequence": 1, "run_id": "...", "created_at": "..."},
    {"role": "assistant", "content": "...", "sequence": 2, "run_id": "...", "created_at": "..."}
  ],
  "pending_actions": []
}
```

`status`：`running` / `paused` / `done` / `error` / `cancelled`。
`pending_actions`：已发出但没有对应 `tool_result` 的工具调用。
非本人会话 / 运行不存在 → 404；v2 持久化未装配 → 503。

### 4.5 `GET /api/v2/chat/events/{run_id}`

断线重连回放。

**Query**

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `from_seq` | `0` | 返回 `sequence > from_seq` 的事件；0 表示全量 |
| `request_id` | `""` | 路径 id 解析失败时的兜底 |

**响应 200**：

```json
{
  "run_id": "...",
  "from_seq": 0,
  "events": [
    {"version": "2", "sequence": 1, "type": "run_started", "payload": {}, "...": "..."}
  ]
}
```

事件元素与 SSE 信封同构。找不到 → 404；非本人 → 404；v2 未装配 → 503。

### 4.6 `POST /api/v2/chat/cancel/{run_id}`

幂等取消。路径参数也接受 `request_id`。

**请求体**：

```json
{"idempotency_key": "cancel-key-1"}
```

**响应**：

| 状态 | 响应体 | 说明 |
| --- | --- | --- |
| 200 | `{"accepted":true,"reason":"cancelled","run_id":"..."}` | 首次取消成功 |
| 409 | `{"accepted":false,"reason":"already_terminal:<type>","run_id":"..."}` | 已结束（done/error/cancelled） |
| 409 | `{"accepted":false,"reason":"unknown_run","run_id":"..."}` | run 不存在 |
| 404 | `{"detail":"Run not found"}` | 非本人会话 |
| 503 | `{"detail":"..."}` | v2 未装配 |

注意：cancel 后流内会收到 `cancelled` 终端；已发出的工具副作用不会回滚。

### 4.7 `POST /api/v2/chat/resume/{run_id}`

审批后恢复被暂停的运行，返回 SSE。**路径必须是真实 `run_id`**（该接口不解析 `request_id`）。

**Query**

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `from_seq` | `0` | 先回放 `sequence > from_seq` 的历史事件，再续跑 |

**行为**：
- 复用原始 checkpoint 和原始 sandbox lease；lease 丢失则 fail-closed，返回 `error/sandbox_lease_lost`，不会新建沙箱。
- 幂等：已完成的步骤（已持久化的 `tool_result` 等）不会重复执行。
- 已终态 run 只回放，不再产生新事件。

**HTTP 状态**：

| 状态 | 场景 | 响应体 |
| --- | --- | --- |
| 200 | 恢复流开始 | SSE |
| 404 | run 不存在 / 非本人 | `{"detail":"Run not found"}` |
| 409 | run 仍是 `paused` | `{"run_id","reason":"run_paused","message":"…先 POST /api/v2/approvals/{approval_id}/decision。"}` |
| 503 | 容量耗尽 | `{"error":{"code":"run_capacity_exceeded" 或 "sandbox_capacity_exceeded","retryable":true}}` |
| 503 | v2 未装配 | `{"detail":"..."}` |

### 4.8 `POST /api/v2/approvals/{approval_id}/decision`

提交高风险工具（如 `sandbox_bash`）的审批结果。该接口同步返回，不产生 SSE；之后客户端自行连接 `resume_url`。

**请求体**：

```json
{"decision": "approve", "decision_request_id": "decision-key-1"}
```

`decision` 只能是 `approve` / `reject`；`decision_request_id` 是幂等键。

**响应 200**：

```json
{
  "approval_id": "...",
  "run_id": "...",
  "tool": "sandbox_bash",
  "decision": "approve",
  "status": "approved",
  "idempotent": false,
  "resume_url": "/api/v2/chat/resume/<run_id>"
}
```

**HTTP 状态**：

| 状态 | 场景 |
| --- | --- |
| 200 | 决策已记录；同 `decision_request_id` 重放返回 `idempotent:true` |
| 404 | approval 不存在 / 非本人 |
| 409 | 已用不同 key 决策过、审批过期、run 已终态、run 不在 paused、sandbox lease 丢失 |
| 422 | `decision` 非法或字段缺失（响应体可能是 `detail` 字符串或列表） |
| 503 | v2 未装配 |

409 响应形如 `{"approval_id","reason","message","current_run_status"?}`。

### 4.9 `GET /api/v2/artifacts/{artifact_id}`

下载 artifact。`artifact_id` 必须是 UUID；否则 404。

**Query**：`conversation_id`（可选，提供时做归属校验）。

**响应 200**：原始字节，`Content-Type` 为存储的 MIME，`Content-Disposition: attachment; filename="<public_filename>"`。

| 状态 | 场景 |
| --- | --- |
| 200 | 文件返回 |
| 403 | `conversation_id` 不匹配 |
| 404 | 非 UUID / 无记录 / 非本人 subject |
| 410 | `expired`、`failed`、已过 `expires_at`、文件缺失、PDF 内容校验不通过 |
| 503 | v2 未装配 |

所有 410 共用 `{"detail":"Artifact unavailable"}`。

### 4.10 `GET /api/v2/memory/settings` / `PUT`

读取 / 修改当前 subject 的记忆总开关。记忆关闭时不会产生新候选，也不会在对话中注入记忆。

`GET` → `{"subject_id":"...","enabled":true}`

`PUT` 请求体：

```json
{"enabled": true}
```

响应同 GET。服务未装配 → 503。

### 4.11 `GET /api/v2/memories`

**Query**：`limit`（默认 100，1–500）、`offset`（默认 0）。

```json
{
  "items": [
    {
      "memory_id": "...",
      "category": "preference",
      "content": "喜欢轻量徒步",
      "created_at": "...",
      "updated_at": "...",
      "expires_at": null,
      "consent": {"type": "candidate_approval", "consented_at": "...", "candidate_id": "..."}
    }
  ],
  "total": 1,
  "limit": 100,
  "offset": 0
}
```

### 4.12 `PATCH /api/v2/memories/{memory_id}`

```json
{"content": "新的内容", "category": "preference", "ttl_days": 30}
```

`content` 1–4000 字符，`category` 1–64 字符（现有取值为 `preference` / `contact` / `health` / `trip` / `location`），`ttl_days` 1–3650 或 `null`（不过期）。成功返回更新后的 memory item；不存在 / 非本人 → 404。

### 4.13 `DELETE /api/v2/memories/{memory_id}`

成功：`{"deleted": true}`；不存在 / 非本人 → 404。

### 4.14 `DELETE /api/v2/memories?confirm=true`

清空当前 subject 全部记忆。缺少 `confirm=true` → 409 `{"detail":"Confirmation required"}`；成功：`{"deleted": <条数>}`。

### 4.15 `GET /api/v2/memories/export`

把活跃记忆导出为 JSON 文件（7 天有效），返回 artifact 句柄：

```json
{"artifact_id": "...", "expires_at": "..."}
```

随后用 `GET /api/v2/artifacts/{artifact_id}` 下载。持久化未装配 → 503。

### 4.16 `POST /api/v2/memory/candidates/{candidate_id}/decision`

对敏感记忆候选做逐条同意 / 拒绝。

```json
{"decision": "approve", "decision_request_id": "cand-key-1"}
```

**响应 200**：

```json
{"candidate_id": "...", "status": "approved", "decided_at": "...", "memory": {"memory_id": "...", "category": "contact", "content": "...", "consent": {"...": "..."}}, "idempotent": false}
```

拒绝时不返回 `memory`。同 key 重放返回 `idempotent:true`；不同 key 重复决策 → 409；不存在 → 404。

### 4.17 `POST /api/v2/subject/rotate`

轮换匿名身份 token，旧 cookie 立即失效，响应会下发新 cookie：

```json
{"subject_id": "...", "rotated": true}
```

轮换后无法再访问旧 subject 的 run / memory / artifact。

### 4.18 `POST /api/v2/sessions/lifecycle`

前端上报会话快照的生命周期事件。需要登录。服务端只写一条结构化日志，不保存快照内容，也不读写聊天数据。

```json
{
  "event": "session.created",
  "reason": "new_session",
  "session_id": "session-...",
  "conversation_id": "chat-...",
  "snapshot_id": "snap-...",
  "outgoing_session_id": "session-...",
  "outgoing_snapshot_id": "snap-...",
  "superseded_snapshot_id": "snap-...",
  "snapshot_bytes": 1234,
  "destroyed_count": 0
}
```

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `event` | 是 | `session.created`、`session.restored`、`session.closed`、`session.deleted`、`session.pruned`、`session.snapshot.failed` |
| `reason` | 是 | `new_session`、`switch`、`window_exit`、`tab_hidden`、`delete`、`startup_prune`、`quota_exceeded`、`storage_unavailable` |
| `session_id` | 是 | 会话 id，1–128 字符，首字符为字母或数字，其余可为字母、数字、下划线或短横线 |
| `conversation_id` / `snapshot_id` / `superseded_snapshot_id` / `outgoing_session_id` / `outgoing_snapshot_id` | 否 | 同一 id 规则 |
| `destroyed_count` | 否 | 0–1000，默认 0 |
| `snapshot_bytes` | 否 | 0–10000000 |

未知字段返回 422。**响应 204**：空响应体。

日志包含 `event`、`session_id`、`conversation_id`、`snapshot_id`、`superseded_snapshot_id`、`outgoing_session_id`、`outgoing_snapshot_id`、`snapshot_bytes`、`destroyed_count` 和当前 `subject_id`；不记录标题、消息正文或工具输出。

### 4.19 `POST /api/v1/models/fetch`

服务端代拉上游模型列表，客户端密钥只在本次请求使用，不落库。

```json
{"base_url": "https://api.example.com/v1", "api_key": "sk-..."}
```

**响应 200**：`{"models":["model-a","model-b"]}`。

| 状态 | 场景 |
| --- | --- |
| 200 | 成功 |
| 4xx/5xx | 上游返回同状态码，`detail` 为 `Upstream returned <code>` |
| 502 | 无法连接上游 |

### 4.20 `GET /api/v1/llm-config`

读取当前生效的模型配置。密钥只返回打码值与“是否已设置”，**永不返回明文**。

```json
{
  "llm": {"base_url": "https://api.edgefn.net/v1", "model": "DeepSeek-V4-Flash-0731", "api_key_masked": "sk-a...wxyz", "api_key_set": true},
  "embedding": {"base_url": "https://api.edgefn.net/v1", "model": "BAAI/bge-m3", "dimensions": 1024, "api_key_masked": "sk-a...wxyz", "api_key_set": true},
  "rerank": {"base_url": "https://api.edgefn.net/v1", "model": "bge-reranker-v2-m3", "api_key_masked": "sk-a...wxyz", "api_key_set": true},
  "updated_at": "2026-10-22T00:48:00+00:00",
  "index_dimensions": 1024
}
```

`updated_at` 为 `.env` 文件修改时间；`index_dimensions` 为 RAG 索引实际向量维度（索引不存在时为 `null`）。

**本机限制**（三个 llm-config 接口共用）：仅允许 client IP 为 `127.0.0.1` / `::1`；带 `Origin` 时必须是本机 http(s) 来源；`VERCEL` 环境下直接 403（`{"detail":"llm-config ..."}`）。Gateway 未注册该路由。

### 4.21 `PUT /api/v1/llm-config`

写入 `ai-service/.env` 并立即热更新运行时配置。**三段配置都必须提供**；`api_key` 省略或传空表示保留已存密钥。

```json
{
  "llm": {"base_url": "https://api.example.com/v1", "model": "model-a", "api_key": "sk-..."},
  "embedding": {"base_url": "https://api.example.com/v1", "model": "bge-m3", "dimensions": 1024, "api_key": ""},
  "rerank": {"base_url": "https://api.example.com/v1", "model": "bge-reranker-v2-m3"}
}
```

**响应 200**：同 GET 的完整快照，外加：

```json
{"reindex_recommended": true}
```

`reindex_recommended=true` 表示 embedding 的 `base_url` 或 `model` 发生变化，建议重建知识库索引。

| 状态 | 场景 |
| --- | --- |
| 400 | 缺少某段配置、`base_url`/`model` 为空、`dimensions` 不在 64–4096、字段非法 |
| 409 | `embedding.dimensions` 与现有索引实际维度不一致（拒绝写入） |
| 500 | `.env` 写入失败，配置未保存 |
| 403 | 非本机来源 / Vercel 环境 |

请求体 `extra="forbid"`，未知字段 → 422。

### 4.22 `POST /api/v1/llm-config/test`

探测某一段配置的上游 `/models`，不落盘。`base_url` / `api_key` 为空时回退已存配置。

```json
{"section": "llm", "base_url": "https://api.example.com/v1", "api_key": "sk-..."}
```

**响应 200**：

```json
{"latency_ms": 320, "model_count": 12, "models": ["model-a", "model-b"]}
```

`model_count` 是上游 `data` 数组长度，`models` 只保留合法 `id`。

| 状态 | 场景 |
| --- | --- |
| 400 | `section` 非法；`base_url` 未配置；请求了与已存不同的 `base_url` 但未带 `api_key`（防止把已存密钥发往新主机） |
| 401 / 403 | 上游拒绝访问（同码透传，`detail` 为「上游拒绝访问」） |
| 502 | 连接失败、上游 >= 400、或响应不是有效 JSON |
| 403 | 非本机来源 / Vercel 环境 |

超时固定 15 秒。

## 5. 错误码（SSE `error.payload.code`）

| code | 含义 |
| --- | --- |
| `v2_disabled` | `AGENT_V2_ENABLED=false` |
| `unknown_profile` | profile 不存在 |
| `cowork_unauthorized` | cowork 未授权给当前部署 |
| `conversation_forbidden` | 会话不属于当前 subject |
| `model_unavailable` | 未配置 OPENAI_API_KEY |
| `model_timeout` | 模型超时 |
| `kernel_model_build_failed` / `kernel_run_failed` / `kernel_unavailable` | 内核调用失败 |
| `empty_model_output` | 模型空输出 |
| `context_unavailable` | 会话历史读取失败 |
| `checkpoint_unavailable` | checkpoint 不可用 |
| `run_not_found` | 运行不存在（resume 流内） |
| `no_resolved_approval` | resume 时找不到已解决审批 |
| `sandbox_creation_failed` / `sandbox_unavailable` / `sandbox_quota` / `sandbox_timeout` / `sandbox_type_invalid` / `sandbox_lease_lost` | 沙箱相关失败 |
| `tool_denied` | 工具被权限门拒绝 |
| `bash_plan_invalid` / `bash_execution_failed` | cowork Bash 失败 |
| `artifact_generation_failed` / `artifact_export_failed` / `artifact_evidence_persistence_failed` | artifact 失败 |
| `effect_in_flight` | 同一副作用正被其他执行者处理 |
| `effect_outcome_unknown` | 副作用结果不确定，需人工检查 |
| `evidence_unavailable` | 检索证据不可用 |
| `internal_error` | 未预期内部错误 |

未在公开清单内的新错误码在日志/trace 中记为 `stream_error`，客户端只应依赖终端语义。

## 6. Capability profile

| profile | 能力 | 默认可用 |
| --- | --- | --- |
| `chat` | 普通徒步对话 + 低风险徒步工具 | 是 |
| `route_plan` | chat + 路线研究、Web、白名单地图/天气 MCP | 是 |
| `artifact` | 独立沙箱文件工具 + PDF 生成，无 Bash / MCP | 是 |
| `cowork` | artifact + 沙箱内 Bash，高风险操作需审批 | 仅 `AGENT_V2_COWORK_POLICY=trusted/staging` |
| `rag` | 知识库问答（`hiking_knowledge_search`），走同一内核 + 证据门 | 是 |

`profile_id` 只是“请求”，服务端按部署策略授权；不能靠传 profile 越权。

## 7. 典型调用流程

### 7.1 普通对话

1. `POST /api/v2/chat/sse`（带持久 cookie jar）
2. 消费事件直到 `done` / `error` / `cancelled`
3. 掉线：记下最后一个 `sequence`，`GET /api/v2/chat/events/{run_id}?from_seq=N` 续传
4. 刷新页面：`GET /api/v2/chat/snapshot/{run_id}` 恢复消息与状态
5. 用户点“停止”：`POST /api/v2/chat/cancel/{run_id}`

### 7.2 高风险操作审批（cowork）

1. `POST /api/v2/chat/sse`，收到 `approval.required` + `run.paused`
2. 用户确认后：`POST /api/v2/approvals/{approval_id}/decision`
3. 打开 `POST /api/v2/chat/resume/{run_id}?from_seq=<last_seq>` 接收续流
4. 续流序列：`approval.resolved` → `tool_call` → `tool_result` →（artifact 流程）`artifact_candidate` → `artifact.available` → `text` → `done`

### 7.3 artifact 下载

1. 从 `artifact.available` 取 `artifact_id`
2. `GET /api/v2/artifacts/{artifact_id}?conversation_id=<可选>` 下载字节
3. 403/404/410 分别按语义处理；410 统一表示“已不可用”

### 7.4 记忆

1. `GET /api/v2/memory/settings` 确认开关
2. 对话中收到 `memory.consent_required` → `POST /api/v2/memory/candidates/{id}/decision`
3. 管理：`GET /api/v2/memories` → `PATCH` / `DELETE`
4. 导出：`GET /api/v2/memories/export` → `GET /api/v2/artifacts/{id}`

## 8. 与运行环境的注意事项

- 服务只读 `ai-service/.env`；仓库根 `.env` 供 docker-compose 使用，不影响服务。
- `AGENT_V2_ENABLED` 必须为 `true`，否则 `/health` 仍 200，但 chat 只返回 `v2_disabled`。
- 单写者：`agent_state/writer.lock` 保证同一时间只有一个 ai-service 实例持有 v2 运行时；第二实例启动会失败。
- `/health` 为 `ready` 不代表模型可用；验收需实发一轮 SSE 并检查 `text` 正文。
