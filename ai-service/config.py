import os
import json
import re
from pathlib import Path

from dotenv import load_dotenv


if not os.getenv("VERCEL"):
    # README and Docker Compose use the repository root as the canonical
    # configuration location. Keep the service-local file as a fallback for
    # developers who run the service from ``ai-service/`` directly; exported
    # environment variables always take precedence.
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    load_dotenv(Path(__file__).with_name(".env"))


class Settings:
    route_decision_provider: str = "llm"
    route_decision_timeout_seconds: float = 15.0
    route_compare_data_mode: str = "live"
    route_compare_demo_scenario: str = "complete"
    route_decision_external_adapter: str = "route_compare.typesafe_adapter:TypeSafeProvider"
    route_decision_external_model: str = "jev-latest"
    route_decision_external_url: str = "https://api.typesafe.ai/v1/systemone"
    typesafe_api_key: str = ""
    openai_base_url: str = ""
    openai_api_key: str = ""
    openai_model: str = "deepseek-v4-flash"
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    rerank_base_url: str = ""
    rerank_api_key: str = ""
    rerank_model: str = "Qwen/Qwen3-Reranker-8B"
    rerank_top_k: int = 4
    rerank_timeout_seconds: float = 15.0
    rerank_enabled: bool = True
    database_url: str = "postgresql://ai_hiking:ai_hiking@localhost:5432/ai_hiking"
    database_url_source: str = ""
    database_connect_timeout_seconds: int = 2
    redis_url: str = "redis://localhost:6379/0"
    redis_url_source: str = ""
    redis_rest_url: str = ""
    redis_rest_url_source: str = ""
    redis_rest_token: str = ""
    redis_rest_token_source: str = ""
    rag_docs_api_url: str = ""
    rag_sqlite_db_path: str = "./data/rag.sqlite"  # SQLite 是唯一检索存储后端
    feishu_app_id: str = ""
    feishu_app_secret: str = ""
    feishu_default_space_id: str = ""
    feishu_default_folder_token: str = ""
    amap_api_key: str = ""
    mcp_servers: dict = {}
    disabled_lanes: str = ""  # Comma-separated lane names to disable, e.g. "SIMPLE_TOOL,WORKFLOW"
    # Ticket 03: Chat v2 (UniHarness) 配置 —— 默认关闭，灰度期由 Gateway 分桶控制
    # agent_v2_enabled=False 时 v1 Agent/Gateway/Frontend/RAG 行为完全不变；
    # VERCEL + agent_v2_enabled=True 同时出现时启动检查失败（见 agent_v2.startup）
    agent_v2_enabled: bool = False
    # UniHarness 不可变 commit pin（Ticket 25：vendored 真实 fork 基线，
    # provenance 见 third_party/uniharness-provenance.md；值为含项目 delta
    # 的 fork HEAD，upstream 基点 757bf3d… 必须是其祖先）
    agent_v2_uniharness_pin: str = "b026a26250487678aedc2beef0d0b6ae802de5c4"
    # v2 持久化目录（chat_checkpoints/agent_control/agent_memory 三个 SQLite 的父目录）
    agent_v2_state_dir: str = "./agent_state"
    # Ticket 09: artifact profile 使用的沙箱类型。
    #   remote_e2b   — 生产环境，强制使用 E2B（fail-closed，失败不回退宿主）
    #   local_native — 开发环境，使用宿主临时目录
    #   wsl_localvm  — 开发环境，使用 WSL 本地虚拟机（暂以 local_native 实现）
    agent_v2_sandbox_type: str = "remote_e2b"
    # Ticket 13: cowork profile 授权策略。匿名生产用户默认不能访问
    # cowork；只有显式配置 ``trusted`` 或 ``staging`` 才允许该 profile。
    #   disabled — 默认。所有 profile_id="cowork" 请求被拒绝（AC2）。
    #   staging  — 预发/内部测试环境启用，仍走完整审批 + effect ledger。
    #   trusted  — 内部受信任主体可使用，仍走完整审批 + effect ledger。
    # 客户端无法通过 profile_id 升级权限；服务端 gate 是权威。
    agent_v2_cowork_policy: str = "disabled"
    # Ticket 23: bounded one-level subagents 总开关。默认关闭；激活需
    # 显式设置 AGENT_V2_SUBAGENT_ENABLED=true 且 Ticket 22 切流后 14 天
    # 稳定观察期已过。代码组件就绪后此 flag 仍是权威 gate；运营可随时
    # 通过此配置立即关闭 subagent（AC7）。
    agent_v2_subagent_enabled: bool = False
    # Langfuse LLM 可观测性（可选）：未配置 key 时自动禁用，零回归。
    langfuse_enabled: bool = True
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"
    ai_service_allowed_origins: str = "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173"

    def _default_runtime_dir(self, name: str) -> str:
        if os.getenv("VERCEL"):
            return str(Path("/tmp") / name)
        return f"./{name}"

    def _first_env(self, *names: str) -> tuple[str, str]:
        for name in names:
            value = os.getenv(name, "").strip()
            if value:
                return value, name
        return "", ""

    _env_ref_pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

    def _expand_env_refs(self, value):
        if isinstance(value, str):
            return self._env_ref_pattern.sub(lambda match: os.getenv(match.group(1), ""), value)
        if isinstance(value, list):
            return [self._expand_env_refs(item) for item in value]
        if isinstance(value, dict):
            return {key: self._expand_env_refs(item) for key, item in value.items()}
        return value

    def _json_env(self, name: str) -> dict:
        value = os.getenv(name, "").strip()
        if not value:
            return {}
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return self._expand_env_refs(parsed) if isinstance(parsed, dict) else {}

    def load(self) -> "Settings":
        self.route_decision_provider = os.getenv("ROUTE_DECISION_PROVIDER", "llm")
        self.route_decision_timeout_seconds = float(os.getenv("ROUTE_DECISION_TIMEOUT_SECONDS", "15"))
        self.route_compare_data_mode = os.getenv("ROUTE_COMPARE_DATA_MODE", "live")
        self.route_compare_demo_scenario = os.getenv("ROUTE_COMPARE_DEMO_SCENARIO", "complete")
        self.route_decision_external_adapter = os.getenv("ROUTE_DECISION_EXTERNAL_ADAPTER", self.route_decision_external_adapter)
        self.route_decision_external_model = os.getenv("ROUTE_DECISION_EXTERNAL_MODEL", "jev-latest")
        self.route_decision_external_url = os.getenv("ROUTE_DECISION_EXTERNAL_URL", self.route_decision_external_url)
        self.typesafe_api_key = os.getenv("TYPESAFE_API_KEY", "")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_model = os.getenv("OPENAI_MODEL", "deepseek-v4-flash")
        self.embedding_base_url = os.getenv("EMBEDDING_BASE_URL", self.openai_base_url)
        self.embedding_api_key = os.getenv("EMBEDDING_API_KEY", self.openai_api_key)
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
        self.rerank_base_url = os.getenv("RERANK_BASE_URL", self.embedding_base_url)
        self.rerank_api_key = os.getenv("RERANK_API_KEY", self.embedding_api_key)
        self.rerank_model = os.getenv("RERANK_MODEL", "Qwen/Qwen3-Reranker-8B")
        self.rerank_top_k = int(os.getenv("RERANK_TOP_K", "4"))
        self.rerank_timeout_seconds = float(os.getenv("RERANK_TIMEOUT_SECONDS", "15"))
        self.rerank_enabled = os.getenv("RERANK_ENABLED", "true").lower() == "true"
        database_url, database_url_source = self._first_env("DATABASE_URL")
        self.database_url = database_url or "postgresql://ai_hiking:ai_hiking@localhost:5432/ai_hiking"
        self.database_url_source = database_url_source if database_url else "local_default"
        database_timeout_fallback = "10" if os.getenv("VERCEL") else "2"
        self.database_connect_timeout_seconds = int(os.getenv("DATABASE_CONNECT_TIMEOUT_SECONDS", database_timeout_fallback))
        redis_url, redis_url_source = self._first_env(
            "REDIS_URL",
            "KV_URL",
            "UPSTASH_REDIS_URL",
        )
        redis_fallback = "" if os.getenv("VERCEL") else "redis://localhost:6379/0"
        self.redis_url = redis_url or redis_fallback
        self.redis_url_source = redis_url_source if redis_url else ("local_default" if redis_fallback else "")
        self.redis_rest_url, self.redis_rest_url_source = self._first_env(
            "UPSTASH_REDIS_REST_URL",
            "KV_REST_API_URL",
        )
        self.redis_rest_token, self.redis_rest_token_source = self._first_env(
            "UPSTASH_REDIS_REST_TOKEN",
            "KV_REST_API_TOKEN",
        )
        self.rag_docs_api_url = os.getenv("RAG_DOCS_API_URL", "")
        # 01 票据：graph runtime 开关
        # 04 票据：并行检索子问题上限
        self.rag_sqlite_db_path = os.getenv("RAG_SQLITE_DB_PATH", self._default_runtime_dir("data/rag.sqlite"))
        self.feishu_app_id = os.getenv("FEISHU_APP_ID", "")
        self.feishu_app_secret = os.getenv("FEISHU_APP_SECRET", "")
        self.feishu_default_space_id = os.getenv("FEISHU_DEFAULT_SPACE_ID", "")
        self.feishu_default_folder_token = os.getenv("FEISHU_DEFAULT_FOLDER_TOKEN", "")
        self.amap_api_key = os.getenv("AMAP_API_KEY", "")
        self.mcp_servers = self._json_env("MCP_SERVERS")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        self.disabled_lanes = os.getenv("DISABLED_LANES", "")
        # Ticket 03：Chat v2 配置加载
        self.agent_v2_enabled = os.getenv("AGENT_V2_ENABLED", "false").lower() == "true"
        self.agent_v2_uniharness_pin = os.getenv(
            "AGENT_V2_UNIHARNESS_PIN", "b026a26250487678aedc2beef0d0b6ae802de5c4"
        )
        self.agent_v2_state_dir = os.getenv(
            "AGENT_V2_STATE_DIR", self._default_runtime_dir("agent_state")
        )
        # Ticket 09: artifact profile sandbox type. Allowed values:
        # remote_e2b (default, production) / local_native / wsl_localvm.
        # Misconfigured values fall back to remote_e2b so production
        # never accidentally picks a host-backed sandbox.
        _sandbox_type = os.getenv("AGENT_V2_SANDBOX_TYPE", "remote_e2b")
        self.agent_v2_sandbox_type = (
            _sandbox_type
            if _sandbox_type in {"remote_e2b", "local_native", "wsl_localvm"}
            else "remote_e2b"
        )
        # Ticket 13: cowork profile 授权策略加载。允许值：
        # disabled（默认）/ staging / trusted。未知值统一回退 disabled
        # 以保证匿名生产用户永远拿不到 cowork。
        _cowork_policy = os.getenv("AGENT_V2_COWORK_POLICY", "disabled").lower()
        self.agent_v2_cowork_policy = (
            _cowork_policy
            if _cowork_policy in {"disabled", "staging", "trusted"}
            else "disabled"
        )
        # Ticket 23: bounded one-level subagents 总开关。默认关闭；
        # AGENT_V2_SUBAGENT_ENABLED=true 显式启用。代码组件就绪但激活
        # 仍需 Ticket 22 切流后 14 天稳定观察期（由运维外部把关）。
        self.agent_v2_subagent_enabled = (
            os.getenv("AGENT_V2_SUBAGENT_ENABLED", "false").lower() == "true"
        )
        # Langfuse：未配置 key 时 observability 模块返回空 callbacks，零回归。
        self.langfuse_enabled = os.getenv("LANGFUSE_ENABLED", "true").lower() == "true"
        self.langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        self.langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        self.langfuse_base_url = os.getenv(
            "LANGFUSE_BASE_URL", "https://cloud.langfuse.com"
        )
        self.ai_service_allowed_origins = os.getenv(
            "AI_SERVICE_ALLOWED_ORIGINS",
            os.getenv(
                "ALLOWED_ORIGINS",
                "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173",
            ),
        )

        return self

    @property
    def redis_connection_mode(self) -> str:
        if self.redis_url:
            return "redis_url"
        if self.redis_rest_url and self.redis_rest_token:
            return "upstash_rest"
        if self.redis_rest_url or self.redis_rest_token:
            return "incomplete_upstash_rest"
        return "none"

    @property
    def redis_configured(self) -> bool:
        return self.redis_connection_mode in {"redis_url", "upstash_rest"}

    @property
    def postgres_configured(self) -> bool:
        return self.database_url_source == "DATABASE_URL"

    @property
    def feishu_enabled(self) -> bool:
        return bool(self.feishu_app_id and self.feishu_app_secret)

    @property
    def feishu_default_configured(self) -> bool:
        return bool(self.feishu_default_space_id or self.feishu_default_folder_token)

    @property
    def cors_origins(self) -> list[str]:
        """Return explicit CORS origins, never a wildcard."""
        return [
            origin.strip()
            for origin in self.ai_service_allowed_origins.split(",")
            if origin.strip() and origin.strip() != "*"
        ]


settings = Settings().load()
