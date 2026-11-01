# AI Hiking _(ai-hiking)_

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)
![CI](https://img.shields.io/github/actions/workflow/status/Vortex745/hiking-ai/deploy.yml?label=CI&logo=githubactions&style=flat-square)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white&style=flat-square)
![Vite](https://img.shields.io/badge/Vite-5-646CFF?logo=vite&logoColor=white&style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white&style=flat-square)
![Go](https://img.shields.io/badge/Go-1.22-00ADD8?logo=go&logoColor=white&style=flat-square)
![Docker Hub](https://img.shields.io/docker/pulls/zijinn123/ai-hiking?label=Docker%20Hub&logo=docker&style=flat-square)

面向户外徒步场景的 RAG 与 Agent 智能助手。

AI Hiking 将 React 前端、Go API Gateway 和 Python FastAPI AI Service 组合成一个本地优先的徒步助手。项目提供知识库问答（chat 内知识检索工具）、路线规划、天气适宜性判断、装备检查、风险提醒、SSE 流式输出、浏览器侧模型配置和会话记忆能力。

仓库名和目录名是 `ai-hiking`；前端 `package.json` 的包名是 `ai-hiking-frontend`，因为这是一个包含前端、网关和 AI 服务的多模块仓库。

## 目录

- [安全](#安全)
- [背景](#背景)
- [安装](#安装)
- [使用](#使用)
- [架构](#架构)
- [页面](#页面)
- [测试](#测试)
- [部署](#部署)
- [排障](#排障)
- [API](#api)
- [维护者](#维护者)
- [贡献](#贡献)
- [许可证](#许可证)

## 安全

不要把 `.env`、API Key、SSH Key、Cookie 或第三方平台 Token 提交到仓库。模型、Embedding、Rerank、高德地图和 Feishu 等密钥应只存在于本地 `.env`、GitHub Secrets 或服务器运行环境中。

`/llm-config` 会把浏览器侧模型配置保存在 localStorage，并随请求发送给后端；它不会把密钥写入仓库。

## 背景

徒步问答通常不只是一次普通聊天：用户可能需要结合知识库资料、天气、当前位置、路线、装备和风险判断。AI Hiking 因此把系统拆成三个清晰的运行层：

- 前端负责聊天界面、模型配置、浏览器定位和 SSE 渲染。
- Gateway 负责统一 `/api/v1/*`（models）与 `/api/v2/*`（Chat）入口、CORS、限流和流式代理。
- AI Service 负责唯一编排内核（Agent v2，含 RAG 问答 profile）、检索基线和徒步领域工具。

主要能力包括：

- RAG：文档加载、向量检索、BM25 词法检索、RRF 融合、可选 rerank。
- Agent（Chat v2）：UniHarness 内核、可恢复运行、工具审批、产物管理和记忆提交。
- Memory（v2）：长期记忆服务与会话存储由 Agent v2 内核统一持有（旧版 memory 包已移除）。

## 安装

### 依赖

需要本机具备：

- Docker 和 Docker Compose
- Node.js 20 或兼容版本
- Python 3.12
- Go 1.22
- 可选：`lark-cli`，用于 Feishu 同步
- 可选：高德地图 MCP Server，用于地理和天气工具
- 会话与长期记忆由 Agent v2 的 SQLite 存储持有（旧版 Redis 聊天历史已随 memory 包移除）

### 克隆仓库

```bash
git clone https://github.com/Vortex745/hiking-ai.git
cd hiking-ai
```

### 配置环境变量

```bash
cp .env.example .env
```

至少需要填写主模型和 Embedding 配置：

```env
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-llm-api-key
OPENAI_MODEL=deepseek-v4-flash

EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=your-embedding-api-key
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

MCP_SERVERS={"amap":{"command":"amap-mcp","args":["--stdio"]}}

可选：配置 Langfuse 追踪 LLM 调用（未配置密钥时自动禁用）：

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```
```

### Docker Compose 启动

Docker Compose 当前仅用于启动基础设施（PostgreSQL 和 Redis，Redis 已不被应用消费，保留仅为兼容）。应用服务请在本地通过环境依赖启动。

```bash
docker compose up -d
```

默认端口：

| 服务 | 地址 |
|---|---|
| PostgreSQL | `localhost:5432` |
| Redis | `localhost:6379` |

## 使用

### 本地开发

启动基础设施：

```bash
docker compose up -d
```

启动 AI Service：

```bash
cd ai-service
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

启动 Gateway：

```bash
cd gateway
go run main.go
```

启动 Frontend：

```bash
cd frontend
npm install
npm run dev
```

开发模式访问：

```text
http://localhost:5173
```

### 账号与数据隔离

首次打开会进入登录页，可直接注册一个本机账号。

- 登录 ID 为 2–6 位字母、数字或下划线，区分大小写且唯一；密码 8–15 个字符。
- JWT 保存在 HttpOnly Cookie 中（同时接受 `Authorization: Bearer`），`AUTH_JWT_SECRET` 未配置时自动生成 `data/auth.key`。
- 会话记录、长期记忆、知识库、自定义 MCP 和模型配置都按登录账号隔离；高德天气/地图的内置定义共享，但 API Key 由各账号在 MCP 接入中自行填写（保存在 `agent_state/users/<user_id>/mcp_builtin.json`，普通 MCP 管理接口只返回是否已配置）。
- 悬浮地图在「设置 → MCP 接入 → 高德地图」填写当前账号的 JSAPI Key 和安全密钥，Web 服务 API Key 供天气/地图工具使用。悬浮地图不再读取全局 `VITE_AMAP_KEY` / `VITE_AMAP_SECURITY_CODE`，也不自动迁移旧全局配置；未配置时提示「请配置高德地图 API」，不加载地图或触发该入口的定位。专用登录接口仅向当前账号返回浏览器所需配置，并禁止缓存；普通 MCP 列表与保存响应不回传密钥。
- 规则前置：本机管理接口仍只接受本机来源，因此远程访问的账号只能聊天，不能管理知识源/MCP/模型。
- 每个新账号（包括首个注册账号）的 MCP、知识库和模型配置均从空配置开始，需要自行配置；不自动认领旧版 `rag.sqlite`、`mcp_servers.json` 或未归属的知识来源。已有账号保存的配置和历史认领数据保留，切回该账号仍可使用。
- 登录账号的模型配置完全自持：未配置即为空，不再继承 `ai-service/.env`；账号配置保存到 `data/users/<user_id>/llm_config.json`，API Key 加密落盘，接口只返回“已设置”而不回传任何密钥片段。`.env` 只服务无账号的旧单机模式与服务器自身配置。

### 常用健康检查

```bash
curl http://localhost:8000/health
curl http://localhost:8080/health
curl http://localhost:3000/
```

## 架构

```text
Browser
  │ HTTP / SSE
  ▼
frontend/   React 18 + TypeScript + Vite + TailwindCSS
  │ /api/v1/* (RAG / models) + /api/v2/* (Chat)
  ▼
gateway/    Go 1.22 + Gin + CORS + proxy
  │ HTTP / SSE proxy
  ▼
ai-service/ Python 3.12 + FastAPI
  ├─ Agent v2: UniHarness kernel — 唯一编排栈，5 个 profile
  │    chat / route_plan / artifact / cowork / rag
  │    durable runs, approvals, sandbox, conversation store
  ├─ RAG: ingestion, SQLite hybrid retrieval (FTS5 + sqlite-vec + RRF),
  │    rerank, Feishu/cloud-docs sync — 检索基线（无自有运行时）
  └─ Tools: knowledge search（内嵌预算检索子图）, gear, risk,
       route, web, PDF, sandbox tools

SQLite（agent_control / chat_checkpoints / agent_memory / kernel_checkpoints / rag）
```

目录结构：

```text
ai-hiking/
├── ai-service/              # FastAPI AI Service
│   ├── agent_v2/            # UniHarness kernel — 唯一编排栈（runs/approvals/memory）
│   ├── api/                 # FastAPI routers（chat v2 / memory / models）
│   ├── rag/                 # 检索基线：摄入 / SQLite 混合检索 / 同步源（供知识工具调用）
│   ├── tools/               # Base and hiking-domain tools
│   └── tests/
├── frontend/                # React + Vite frontend
│   ├── src/
│   └── tests/
├── gateway/                 # Go API Gateway
│   ├── handler/
│   └── middleware/
├── mcp-server/              # Optional MCP extensions
├── docker-compose.yml
├── docker-compose.prod.yml
└── README.md
```

## 页面

| 路由 | 页面 | 用途 |
|---|---|---|
| `/` | Agent 模块 | 徒步助手对话、工具过程、路线/装备/风险任务 |
| `/llm-config` | 模型配置 | 浏览器侧 LLM、Embedding、Rerank 参数配置 |

## 测试

CI 定义在 [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)，工作流名称为 `CI`。每次分支推送、PR、合并队列和手动触发都会运行。三个服务并行检查：

| 检查 | 内容 |
|---|---|
| Frontend build and browser regression | Node.js 22，`npm ci`、TypeScript 检查、Vite 生产构建、Chromium 会话切换回归 |
| Gateway vet, test and build | 按 `gateway/go.mod` 设置 Go，依赖校验、`go vet`、`go test`、编译 |
| Backend and evaluation contracts (offline) | Python 3.12，离线恢复并核验 fork、联合安装依赖、`pip check`、后端全量离线回归、评估契约 |

`Validate services` 汇总三个检查，只有全部成功才通过；失败、取消或跳过均不能放行。可在 GitHub 分支保护中将其设为 required status check。新提交会取消同一 ref 上的旧运行，任务设有超时，并缓存 npm、Go 和 pip 下载。

Actions 产物保留 7 天：`frontend-dist` 为前端构建，`frontend-browser-results` 包含浏览器日志、预览服务日志及成功截图，`backend-test-results` 包含后端与评估的 JUnit XML。后端回归失败时，已成功安装依赖的评估契约仍会执行并独立记录结果。

本地使用以下命令复现各服务检查。Python 测试先在终端设置离线测试环境（以下为 Bash；PowerShell 使用对应的 `$env:变量名 = "值"`）：

```bash
export OPENAI_API_KEY=ci-placeholder-not-a-credential
export KERNEL_LIVE_E2E=0 HIKING_EVAL_LIVE=0
export LANGFUSE_ENABLED=false DEEPEVAL_TELEMETRY_OPT_OUT=YES
```

1. 恢复受控 UniHarness fork（离线 bundle，无网络协议）：

```bash
python scripts/restore-uniharness-fork.py
python scripts/restore-uniharness-fork.py --check
```

2. 安装运行、测试与评估依赖（Python 3.12），并校验依赖一致性：

```bash
python -m pip install -r ai-service/requirements-test.txt -e third_party/uniharness/libs/uniharness -r evals/deepeval/requirements.txt
python -m pip check
```

3. AI Service 离线后端回归（服务目录内，包含 kernel 身份测试与失败注入自检）：

```bash
cd ai-service
python -m pytest -q -rs
```

4. 评估契约（仓库根目录，纯离线，不调用外部模型或 Judge）：

```bash
python -m pytest -q evals/test_evaluation_contract.py
```

Frontend：

```bash
cd frontend
npm ci
npm run build
npx playwright install chromium
```

在一个终端启动生产预览：

```bash
npm run preview -- --host 127.0.0.1 --port 4173 --strictPort
```

在另一个终端的 `frontend/` 下执行浏览器回归（API/SSE 使用测试数据，无需启动后端）：

```bash
BASE_URL=http://127.0.0.1:4173 npm run test:browser
```

Gateway：

```bash
cd gateway
go mod download
go mod verify
go vet ./...
go test ./...
go build -trimpath -o ../test-report/gateway .
```

### 评分状态与结论范围

- 评估结果只有 `passed` / `failed` / `not_evaluated` 三种状态；未执行或无法判定的评估记为 `not_evaluated`，绝不按通过处理（语义见 `evals/deepeval/README.md`）。
- 默认验证 job 完全离线：不请求外部模型，不需要 OPENAI / Jev 凭证；无凭证的在线契约以 recorded skip 呈现，不会被计为通过。
- 在线模型契约与在线质量评估保留显式 opt-in：后端在线契约需 `KERNEL_LIVE_E2E=1` 加真实 `OPENAI_*` 凭证；在线质量评估需 `HIKING_EVAL_LIVE=1` 加 Jev 凭证（见 `evals/deepeval/README.md`）。
- 离线契约通过只证明离线行为契约（远程副作用拒绝、事实门禁、真实 PDF 与非法下载拒绝、有界多轮历史、依赖可复现、评估状态如实上报），不代表线上模型质量已达标。
- 本仓库不沿用其他工作树的历史通过计数；未实际执行远程 GitHub Actions 或在线质量评估时，不得宣称已完成对应验证。

## 部署

### GitHub Actions

`.github/workflows/deploy.yml` 沿用历史文件名，目前执行上文的 CI 检查。运行 CI 无需配置 GitHub Secrets；其中的模型 Key 是构造测试对象用的占位值。生产发布通过下述部署方式单独进行。

### 生产 Compose

```bash
DOCKERHUB_IMAGE=zijinn123/ai-hiking IMAGE_TAG=latest docker compose -f docker-compose.prod.yml up -d
```

生产 Compose 只包含应用服务和 Postgres。会话与长期记忆由 Agent v2 的 SQLite 存储持有（需要为实例配置持久卷），不再依赖外部 Redis。

### CloudBase

前端生产环境默认 API 地址配置在 `frontend/src/api/config.ts`：

```text
https://gateway-262534-6-1364947792.sh.run.tcloudbase.com/api/v1
```

当前 CloudBase 资源：

| 服务 | 类型 | URL |
|---|---|---|
| Frontend | 静态托管 | `https://ai-hiking-d4gf29b4zcebc048d-1364947792.tcloudbaseapp.com/` |
| Gateway | CloudRun | `https://gateway-262534-6-1364947792.sh.run.tcloudbase.com` |
| AI Service | CloudRun | `https://ai-service-262534-6-1364947792.sh.run.tcloudbase.com` |

## 排障

### Agent 提示缺少 API Key

检查 `.env` 中的 `OPENAI_API_KEY`，或在 `/llm-config` 中配置可用模型参数。

### RAG 没有检索结果

检查：

- `EMBEDDING_API_KEY` 是否可用。
- PostgreSQL 是否运行。
- 知识库中是否已经灌入文档（RAG 独立页已删除，前端不再有上传入口）。
- `DATABASE_URL` 是否指向正确数据库。

### 天气或位置工具不可用

配置：

```env
MCP_SERVERS={"amap":{"command":"amap-mcp","args":["--stdio"]}}
```

### Feishu 同步不可用

确认 `lark-cli` 已安装并可在 `PATH` 中访问；必要时设置 `FEISHU_DEFAULT_SPACE_ID` 和 `FEISHU_DEFAULT_FOLDER_TOKEN`。

## API

前端访问 Gateway 的 `/api/v1/*`（models）与 `/api/v2/*`（Chat，唯一 chat 链路）。
RAG 能力经 chat 内的知识检索工具（`hiking_knowledge_search`）提供，没有独立 HTTP 接口（A11 删除）。

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | Gateway 健康检查 |
| `POST` | `/api/v1/models/fetch` | 拉取模型列表 |
| `POST` | `/api/v2/chat/sse` | SSE Chat v2 对话（唯一 chat 链路） |
| `GET` | `/api/v2/chat/snapshot/:run_id` | 运行快照（刷新恢复） |
| `POST` | `/api/v2/chat/cancel/:run_id` | 取消运行 |
| `GET` | `/api/v2/chat/events/:run_id` | 增量事件拉取（from_seq） |
| `POST` | `/api/v2/chat/resume/:run_id` | 恢复运行（审批后继续） |
| `POST` | `/api/v2/approvals/:approval_id/decision` | 工具审批决策 |
| `GET`/`PUT` | `/api/v2/memory/settings` | 记忆开关配置 |
| `GET`/`DELETE` | `/api/v2/memories` | 长期记忆列表 / 清空 |
| `GET` | `/api/v2/memories/export` | 导出长期记忆 |
| `PATCH`/`DELETE` | `/api/v2/memories/:memory_id` | 编辑 / 删除单条记忆 |
| `POST` | `/api/v2/memory/candidates/:candidate_id/decision` | 记忆候选确认 |
| `GET` | `/api/v2/artifacts/:artifact_id` | 下载运行产物 |
| `POST` | `/api/v2/subject/rotate` | 轮换匿名 subject 标识 |

## 维护者

[@Vortex745](https://github.com/Vortex745)

## 贡献

欢迎提交 Issue 和 Pull Request。贡献前请先运行相关模块测试，并避免提交密钥、生成文件、运行时数据或与变更无关的重构。

问题反馈入口：

- [GitHub Issues](https://github.com/Vortex745/hiking-ai/issues)
- [Pull Requests](https://github.com/Vortex745/hiking-ai/pulls)

## 许可证

UNLICENSED © Vortex745
