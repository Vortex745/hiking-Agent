# AI Hiking _(ai-hiking)_

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)
![Deploy](https://img.shields.io/github/actions/workflow/status/Vortex745/hiking-ai/deploy.yml?branch=main&label=deploy&logo=githubactions&style=flat-square)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white&style=flat-square)
![Vite](https://img.shields.io/badge/Vite-5-646CFF?logo=vite&logoColor=white&style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white&style=flat-square)
![Go](https://img.shields.io/badge/Go-1.22-00ADD8?logo=go&logoColor=white&style=flat-square)
![Docker Hub](https://img.shields.io/docker/pulls/zijinn123/ai-hiking?label=Docker%20Hub&logo=docker&style=flat-square)

面向户外徒步场景的 RAG 与 Agent 智能助手。

AI Hiking 将 React 前端、Go API Gateway 和 Python FastAPI AI Service 组合成一个本地优先的徒步助手。项目提供知识库问答、文档上传、Feishu 同步、路线规划、天气适宜性判断、装备检查、风险提醒、SSE 流式输出、浏览器侧模型配置和会话记忆能力。

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

- 前端负责聊天界面、文档上传、模型配置、浏览器定位和 SSE 渲染。
- Gateway 负责统一 `/api/v1/*` 入口、CORS、限流和流式代理。
- AI Service 负责 Agent、RAG、Memory 和徒步领域工具。

主要能力包括：

- RAG：文档加载、查询改写、向量检索、BM25、RRF 融合、可选 rerank 和答案增强。
- Agent：意图识别、槽位抽取、按场景选择工具、LangGraph fallback、工具风险分级和记忆提交。
- Memory：聊天历史、会话压缩、知识抽取和回答完成后的长期记忆写入。

## 安装

### 依赖

需要本机具备：

- Docker 和 Docker Compose
- Node.js 20 或兼容版本
- Python 3.12
- Go 1.22
- 可选：`lark-cli`，用于 Feishu 同步
- 可选：高德地图 MCP Server，用于地理和天气工具
- 生产可选：Vercel Marketplace Redis / Upstash，用于持久化聊天历史

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
```

### Docker Compose 启动

Docker Compose 当前仅用于启动基础设施（PostgreSQL 和 Redis）。应用服务请在本地通过环境依赖启动。

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

### 常用健康检查

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/chat/health
curl http://localhost:8000/api/v1/rag/health
curl http://localhost:8000/api/v1/tools/health
curl http://localhost:8080/health
curl http://localhost:3000/
```

## 架构

```text
Browser
  │ HTTP / SSE
  ▼
frontend/   React 18 + TypeScript + Vite + TailwindCSS
  │ /api/v1/*
  ▼
gateway/    Go 1.22 + Gin + CORS + proxy
  │ HTTP / SSE proxy
  ▼
ai-service/ Python 3.12 + FastAPI
  ├─ Agent: intent intake, hiking workflows, LangGraph fallback, memory commit
  ├─ RAG: loader, query rewrite, vector/BM25 retrieval, rerank, answer augment
  ├─ Memory: chat history, compression, knowledge extraction, vector store
  └─ Tools: search, file, terminal, PDF, geo, weather, route, gear, risk

PostgreSQL + pgvector
Redis / Upstash Redis
```

目录结构：

```text
ai-hiking/
├── ai-service/              # FastAPI AI Service
│   ├── agent/               # Agent intake, policy, execution
│   ├── api/                 # FastAPI routers
│   ├── memory/              # Chat and long-term memory
│   ├── rag/                 # Retrieval-augmented generation
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
| `/` | 首页 | 项目入口和模块跳转 |
| `/hiking-rag` | RAG 模块 | 文档上传、知识库问答、检索过程展示 |
| `/super-agent` | Agent 模块 | 徒步助手对话、工具过程、路线/装备/风险任务 |
| `/llm-config` | 模型配置 | 浏览器侧 LLM、Embedding、Rerank 参数配置 |

## 测试

Frontend：

```bash
cd frontend
npm test
npm run build
```

Gateway：

```bash
cd gateway
go test ./...
```

AI Service：

```bash
cd ai-service
python -m compileall -q .
python -m pytest tests -q
```

## 部署

### GitHub Actions

`main` 分支推送会触发 `.github/workflows/deploy.yml`：

1. 校验前端、Gateway、AI Service 语法和生产 Compose 文件。
2. 构建 `ai-service`、`gateway`、`frontend` 三个镜像。
3. 推送到 Docker Hub 仓库 `zijinn123/ai-hiking`。
4. 发布 pgvector 的 runtime 镜像副本。
5. 通过 SSH 上传 `docker-compose.prod.yml` 和生产 `.env`。
6. 在服务器上拉取 `sha-*` 镜像并重启服务。

需要的 GitHub Secrets：

| Secret | 说明 |
|---|---|
| `DOCKERHUB_USERNAME` | Docker Hub 用户名 |
| `DOCKERHUB_TOKEN` | Docker Hub Token |
| `DEPLOY_HOST` | 服务器地址 |
| `DEPLOY_USER` | SSH 用户 |
| `DEPLOY_SSH_KEY` | SSH 私钥 |
| `DEPLOY_PORT` | SSH 端口 |
| `DEPLOY_PATH` | 服务器部署目录 |
| `OPENAI_API_KEY` | 生产 LLM Key |
| `EMBEDDING_API_KEY` | 生产 Embedding Key |
| `RERANK_API_KEY` | 生产 Rerank Key |
| `REDIS_URL` | 可选，外部托管 Redis 连接串 |
| `KV_URL` / `KV_REST_API_URL` / `KV_REST_API_TOKEN` | 可选，Vercel Marketplace Redis / Upstash 自动注入变量 |

### 生产 Compose

```bash
DOCKERHUB_IMAGE=zijinn123/ai-hiking IMAGE_TAG=latest docker compose -f docker-compose.prod.yml up -d
```

生产 Compose 只包含应用服务和 Postgres；Redis 不再作为生产容器启动。需要 Redis-backed chat memory 时，提供外部托管 Redis：

```env
REDIS_URL=rediss://default:...@...
```

在 Vercel 上，优先通过 Marketplace 连接 Upstash for Redis 到 `ai-hiking-ai-service` 项目。AI Service 会自动识别 `REDIS_URL`、`KV_URL`、`UPSTASH_REDIS_URL`，也兼容 `KV_REST_API_URL` + `KV_REST_API_TOKEN` 或 `UPSTASH_REDIS_REST_URL` + `UPSTASH_REDIS_REST_TOKEN`。没有这些变量时，Vercel Functions 会回退到 `/tmp` 文件内存，不能保证跨冷启动持久。

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
- 是否已经在 `/hiking-rag` 上传文档。
- `DATABASE_URL` 是否指向正确数据库。

### 天气或位置工具不可用

配置：

```env
MCP_SERVERS={"amap":{"command":"amap-mcp","args":["--stdio"]}}
```

### Feishu 同步不可用

确认 `lark-cli` 已安装并可在 `PATH` 中访问；必要时设置 `FEISHU_DEFAULT_SPACE_ID` 和 `FEISHU_DEFAULT_FOLDER_TOKEN`。

## API

前端统一访问 Gateway 的 `/api/v1/*`。

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | Gateway 健康检查 |
| `GET` | `/api/v1/chat/health` | Agent 健康检查 |
| `POST` | `/api/v1/chat/sync` | 同步 Agent 对话 |
| `POST` | `/api/v1/chat/sse` | SSE Agent 对话 |
| `GET` | `/api/v1/chat/history/:chatId` | 聊天历史 |
| `POST` | `/api/v1/chat/confirm` | 工具确认 |
| `GET` | `/api/v1/chat/pending/:chatId` | 待确认工具 |
| `POST` | `/api/v1/models/fetch` | 拉取模型列表 |
| `GET` | `/api/v1/rag/health` | RAG 健康检查 |
| `POST` | `/api/v1/rag/upload` | 上传知识库文档 |
| `POST` | `/api/v1/rag/query` | SSE RAG 问答 |
| `GET` | `/api/v1/rag/documents` | 文档列表 |
| `POST` | `/api/v1/rag/feishu/sync` | 同步单个 Feishu 文档 |
| `POST` | `/api/v1/rag/feishu/default-sync` | 同步默认 Feishu 源 |
| `GET` | `/api/v1/tools` | 工具注册表 |
| `GET` | `/api/v1/tools/health` | 工具配置健康检查 |

## 维护者

[@Vortex745](https://github.com/Vortex745)

## 贡献

欢迎提交 Issue 和 Pull Request。贡献前请先运行相关模块测试，并避免提交密钥、生成文件、运行时数据或与变更无关的重构。

问题反馈入口：

- [GitHub Issues](https://github.com/Vortex745/hiking-ai/issues)
- [Pull Requests](https://github.com/Vortex745/hiking-ai/pulls)

## 许可证

UNLICENSED © Vortex745
