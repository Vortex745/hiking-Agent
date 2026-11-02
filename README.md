<div align="center">

# Hiking AI

**徒步 AI 助手：把路线资料、天气、装备和风险信息整理成可核对的徒步计划。**

[快速开始](#快速开始) · [功能](#功能) · [技术结构](#技术结构) · [测试](#测试)

</div>

Hiking AI 面向日常徒步和多日徒步行程规划。你可以用自然语言描述目的地、日期、同行者和偏好；助手会结合路线资料、天气、装备建议与风险信息，帮助比较路线并整理行程。资料不足时会标出待确认项，不把缺失信息当成事实。

它是徒步行程规划助手，不是登山向导，也不能替代官方公告、当地管理部门信息或现场判断。

## 功能

- **徒步路线比较**：汇总候选路线的里程、爬升、下撤点等可查事实，展示来源、证据缺口和风险状态。
- **行程规划**：根据用户提供的日期、集合、交通、住宿和预算等信息整理计划；未提供的内容保留待确认。
- **装备与风险建议**：按天数、季节、经验和露营需求生成清单，并结合路线与天气给出保守建议。
- **天气与地图**：可接入高德 MCP 天气、地图和地点查询；地图展示需配置对应账号的 JSAPI 凭据。
- **徒步知识检索**：从已配置的知识源检索相关内容，并在可用时展示引用来源。
- **计划书导出**：将已确认的路线和行程内容导出为 PDF。
- **对话与账号**：支持流式对话、执行过程查看、账号隔离配置和可选的长期偏好记忆。

外部模型、地图和知识源能力取决于相应服务及账号配置。未配置时，对应功能不可用或结果会明确显示资料不足。

## 快速开始

### 环境要求

- Python 3.12
- Node.js 18 或更高版本
- 可调用的 OpenAI 兼容模型 API

### 1. 获取代码并配置模型

```bash
git clone https://github.com/Vortex745/hiking-Agent.git
cd hiking-Agent
cp .env.example .env
```

PowerShell 可用 `Copy-Item .env.example .env` 替代 `cp`。在 `.env` 中填写 `OPENAI_API_KEY`；并按所用服务填写 `OPENAI_BASE_URL` 和 `OPENAI_MODEL`。向量检索需要可用的 Embedding 服务。高德和其他 MCP 密钥在登录后按账号配置，不要提交真实密钥。

### 2. 安装并启动后端

```bash
cd ai-service
python -m venv .venv
```

激活虚拟环境：

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

安装后端和仓库附带的 UniHarness 依赖，再启动 API：

```bash
python -m pip install -r requirements.txt
python -m pip install -e ../third_party/uniharness/libs/uniharness
python main.py
```

后端默认监听 `http://localhost:8000`。

### 3. 安装并启动前端

在另一个终端，从仓库根目录运行：

```bash
cd frontend
npm ci
npm run dev
```

打开 `http://localhost:5173`。Vite 开发服务器会将 `/api` 请求代理到本机后端。

## 技术结构

| 模块 | 主要技术 |
| --- | --- |
| 前端 | React、TypeScript、Vite、assistant-ui、Tailwind CSS |
| API 与智能体 | FastAPI、LangGraph、仓库维护的 UniHarness fork |
| 知识检索 | SQLite 向量检索与混合检索；可选 PostgreSQL + pgvector |
| 外部工具 | MCP、高德服务、路线研究、装备与风险工具 |
| 导出 | 经过校验与审批的 PDF 计划书 |

本地前后端的启动入口分别是 `ai-service/main.py` 和 `frontend`。`docker-compose.yml` 提供 PostgreSQL 与 Redis 基础设施；它本身不启动完整应用。

## 配置

- 模型和服务端环境变量模板：[.env.example](.env.example)
- 高德 MCP 的 Web 服务 Key 与地图 JSAPI 凭据按账号配置，不要放入前端构建变量或提交到仓库。
- 默认本地开发可使用 SQLite。需要 PostgreSQL、Redis 或远程沙盒时，再配置对应服务地址和凭据。
- 仓库未提供 `LICENSE` 文件，当前没有声明开源许可。

## 测试

```bash
# 前端类型检查与生产构建
cd frontend
npm ci
npm run build

# 后端测试
cd ../ai-service
python -m pip install -r requirements-test.txt
python -m pytest -q
```

前端浏览器回归用例可在 `frontend` 目录运行，例如 `npm run test:browser:auth`、`npm run test:browser:map` 和 `npm run test:browser:plan`。GitHub Actions 还会验证 Gateway 与离线评测契约。

## 相关文档

- [API 参考](docs/api-reference.md)
- [PDF 计划书格式](docs/plan-pdf-format.md)
- [路线比较演示说明](docs/route-comparison-demo.md)
