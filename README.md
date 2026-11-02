<div align="center">
<h1>Hiking AI</h1>

<p><strong>智能登山规划助手</strong><br/>基于 LangGraph 与 RAG 的登山行程生成与安全评估</p>

[![](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=ffffff)](./ai-service)
[![](https://img.shields.io/badge/React_18-61DAFB?style=flat&logo=react&logoColor=000000)](./frontend)
[![](https://img.shields.io/badge/Docker_Compose-2496ED?style=flat&logo=docker&logoColor=ffffff)](./docker-compose.yml)

[快速开始](#快速开始) · [技术栈](#技术栈) · [核心功能](#核心功能) · [配置说明](#配置说明)

</div>


## 简介

登山AI 是一个智能登山规划平台，将 **RAG 知识检索**、**LangGraph 智能体编排** 与 **沙盒化工具执行** 整合进统一对话界面。用户描述登山需求，智能体调用路线研究、天气查询、装备清单、风险评估等工具，生成带引用来源与可视化地图的行程方案。

**核心特性**：类 ChatGPT 对话界面 · 流式推理展示 · 路线候选对比 · 高德地图展示 · PDF 行程导出 · 账号隔离 · MCP 工具扩展

导航：[接口文档](./docs/api-reference.md) | [PDF 排版规范](./docs/plan-pdf-format.md) | [架构设计](./docs/)

---

## 技术栈

| 层 | 技术 |
| --- | --- |
| 前端 | React 18 · Vite 5 · TypeScript · assistant-ui · Tailwind CSS |
| 后端 | FastAPI · LangGraph 1.2.6 · UniHarness（fork） |
| 存储 | SQLite（默认）· PostgreSQL + pgvector（可选）· Redis |
| 检索 | 向量检索 · 混合检索 · 重排模型 |
| 工具 | 路线研究 · 高德天气/地图 MCP · 装备清单 · 风险评估 · PDF 生成 |
| 沙盒 | E2B（生产）· 本地临时目录（开发） |
| 部署 | Docker Compose |

## 快速开始

**前置要求**：Python 3.12、Node.js 18+、Docker Compose（可选）、兼容 OpenAI 接口的模型 API

**1. 克隆代码**

```bash
git clone https://github.com/Vortex745/hiking-Agent.git
cd hiking-ai
```

**2. 配置环境**

```bash
cp .env.example .env
```

最少配置（填入 `.env`）：

```bash
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4-turbo

# 本地 MCP 管理需要 local_native 沙箱
AGENT_V2_SANDBOX_TYPE=local_native
```

**3. 启动后端**

```bash
cd ai-service
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
python main.py  # 监听 :8000
```

**4. 启动前端**

```bash
cd frontend
npm install
npm run dev  # 监听 :5173，代理 /api → :8000
```

浏览器打开 `http://localhost:5173`。

**5. Docker 部署（可选）**

```bash
docker compose up -d  # 仅 PostgreSQL + Redis
# 完整应用栈见 docker-compose.prod.yml
```

> 💡 向量化与重排模型可复用 `OPENAI_*` 配置；知识库检索需可用的嵌入服务，未配置时对话可用但检索能力受限。

---

## 核心功能

### 智能体编排

UniHarness + LangGraph 驱动的智能体内核（`ai-service/agent_v2/kernel.py`），根据用户请求调度工具、检索知识、执行代码，生成结构化回复。支持流式推理展示、执行步骤回放、断点恢复。

**子智能体（实验性，默认关闭）**：三个固定角色可由根智能体派发处理复杂任务：
- `researcher`：收集登山资料（知识检索、路线研究、网页抓取）
- `route_planner`：生成路线方案（装备清单、高德天气/地图 MCP）
- `artifact_writer`：产出结构化文档（沙盒文件操作、PDF 生成）

子智能体采用单层架构、严格工具白名单与并发限制（每次运行最多 2 个），默认关闭（`.env` 中 `AGENT_V2_SUBAGENT_ENABLED=false`），本地开发可修改测试。

### 工具生态

**领域工具**：
- `route_research`：路线候选搜索，提取里程、爬升、下撤点
- `gear_checklist`：根据天数、季节、露营需求生成装备清单
- `risk_assessment`：分析目的地、天气、经验等维度的风险等级
- `hiking_knowledge_search`：检索登山知识库（向量/混合检索 + 重排）

**MCP 服务（账号隔离）**：
- `amap_weather`：高德天气查询
- `amap_map`：路线规划、POI 检索、地理编码

MCP 密钥按账号隔离存储（`data/users/{user_id}/mcp_builtin.json`），需登录后在前端配置，详见 [MCP 配置](#mcp-配置)。

**沙盒工具**：
- `sandbox_bash` / `sandbox_read` / `sandbox_write` / `sandbox_edit`：隔离沙盒内文件与脚本执行
- `generate_pdf`：PDF 行程生成（含排版校验）

### 知识检索

`ai-service/rag/` 提供向量检索与混合检索（关键词 + 语义），接重排模型提升排序质量。知识库存储为 SQLite（默认）或 PostgreSQL + pgvector，切换嵌入模型需重建索引。

### 产物交付

- **路线对比**：候选路线评分与排序（LLM / TypeSafe System One / 离线演示数据）
- **行程 PDF**：结构化行程导出，含目的地、路线、装备、风险评估与应急联系方式
- **地图展示**：高德地图渲染路线与途经点（需配置 JSAPI 密钥）

### 认证与多租户

JWT 登录态（默认有效期 7 天），数据面接口需鉴权。账号配置（MCP 密钥、模型配置、知识源）隔离存储，支持单机匿名模式（遗留兼容）。

---

## 配置说明

### 环境变量

关键配置见 `.env.example`（73 项），按用途分组：

| 分组 | 关键变量 |
| --- | --- |
| 对话模型 | `OPENAI_API_KEY` · `OPENAI_BASE_URL` · `OPENAI_MODEL` |
| 向量化 | `EMBEDDING_MODEL` · `EMBEDDING_DIMENSIONS`（默认复用 `OPENAI_*`） |
| 重排 | `RERANK_MODEL` · `RERANK_ENABLED`（可选，提升检索质量） |
| 数据库 | `DATABASE_URL`（PostgreSQL）· `REDIS_URL`（默认本地） |
| 认证 | `AUTH_JWT_SECRET`（未配置时自动生成）· `AUTH_TOKEN_TTL_SECONDS` |
| 智能体 | `AGENT_V2_ENABLED` · `AGENT_V2_SANDBOX_TYPE` · `AGENT_V2_SUBAGENT_ENABLED` |
| 可观测性 | `LANGFUSE_ENABLED` · `LOG_LEVEL` · `LOG_FORMAT` |

> ⚠️ `.env` 中的 `AMAP_API_KEY` 已废弃，MCP 密钥需在前端配置。

### MCP 配置

高德地图 MCP 服务按账号隔离，需登录后在前端配置：

**密钥类型**：
- **Web 服务 API Key**：智能体工具调用（天气、路线规划、POI）
- **JSAPI Key + 安全密钥**：前端地图展示（可选）

**配置步骤**：
1. 启动应用并登录
2. 前端右上角 → MCP 工具 → 找到「高德天气」或「高德地图」
3. 填写密钥（申请地址：[高德开放平台](https://console.amap.com/dev/key/app)）
4. 保存并验证状态变为「已配置」

**限制**：
- 本地开发限定（`AGENT_V2_SANDBOX_TYPE=local_native`）
- 生产环境（`remote_e2b`）需服务端预置凭据
- 密钥存储路径：`data/users/{user_id}/mcp_builtin.json`

### 智能体定制

- `ai-service/agent_v2/subagents.py`：增改子智能体角色与工具白名单
- `ai-service/tools/`：注册新工具
- `ai-service/agent_v2/profiles.py`：调整模型档位与工具面
- `ai-service/config.py`：读取新增配置项

---

## 测试与评估

### 后端测试

```bash
cd ai-service
pytest -q  # 101 个测试用例
```

### 评测

**离线契约评测**（不调用外部模型，CI 每次 PR 都会跑）：

```bash
pip install -r evals/deepeval/requirements.txt
python -m pytest -q evals/test_evaluation_contract.py
```

**真实模型评测**（opt-in，消耗额度）：

```bash
HIKING_EVAL_LIVE=1 TYPESAFE_API_KEY="..." \
  ./ai-service/.venv/Scripts/deepeval.exe test run evals/deepeval/test_hiking_agent.py
```

### 前端浏览器用例

```bash
cd frontend
npm run test:browser          # 会话切换
npm run test:browser:plan     # 行程工作流
npm run test:browser:auth     # 登录
```

---

## 部署

### 生产部署

`docker-compose.prod.yml` 使用预构建镜像：

```bash
export DOCKERHUB_IMAGE=zijinn123/ai-hiking
export IMAGE_TAG=latest
export FRONTEND_PORT=3000

docker compose -f docker-compose.prod.yml up -d
```

编排包含四个服务：`postgres`（15432）、`ai-service`（容器内 8000）、`gateway`（容器内 8080）、`frontend`（3000 → 容器 80）。

确保 `.env` 配置生产密钥（`AUTH_JWT_SECRET`、模型凭据）、数据库持久卷、Redis 密码。

---

## 已知限制

- 默认 SQLite 存储；Compose 提供 PostgreSQL + pgvector，两条路径需按需对齐
- 切换嵌入模型必须重建知识库索引（维度绑定）
- 远程沙盒默认 `remote_e2b`，需对应凭据；离线环境改为 `local_native`
- `docker-compose.yml` 仅含基础设施（PostgreSQL + Redis），不含应用服务
- Windows 长路径支持需 `git config --global core.longpaths true`

---

## 参考资源

- [接口文档](./docs/api-reference.md)
- [行程 PDF 排版规范](./docs/plan-pdf-format.md)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [FastAPI 官方指南](https://fastapi.tiangolo.com/)

---

## 许可证

仓库当前未包含 `LICENSE` 文件。若计划开源分发，请补充许可证；在此之前默认保留所有权利。

---

<div align="center">

**如果这个项目对您有帮助，请给我们一个 ⭐️**

</div>
