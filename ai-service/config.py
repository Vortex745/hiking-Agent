import os
import json
import re
from pathlib import Path

from dotenv import load_dotenv


if not os.getenv("VERCEL"):
    load_dotenv(Path(__file__).with_name(".env"))


class Settings:
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
    rag_enable_clarification: bool = False  # 08 票据：澄清双请求 + SSE 契约开关（默认关闭，零回归）
    rag_use_graph_runtime: bool = False  # 01 票据：生产 graph runtime 开关（默认关闭，零回归；02-05 票据逐步切换）
    rag_max_sub_questions: int = 3  # 04 票据：并行检索子问题上限（默认 3，零回归）
    # 10 票据：pgvector → SQLite 存储后端 cutover 开关（默认 pgvector，零回归）
    # 12 票据：新增 "memory" 显式后端选项（RAG_STORAGE_BACKEND=memory）
    rag_storage_backend: str = "pgvector"  # "pgvector" | "sqlite" | "memory"
    rag_sqlite_db_path: str = "./data/rag.sqlite"
    rag_cutover_marker_path: str = "./data/cutover-marker.json"
    rag_migration_progress_path: str = "./data/migration-progress.json"
    rag_baseline_performance_path: str = "./data/baseline-performance.json"
    feishu_app_id: str = ""
    feishu_app_secret: str = ""
    feishu_default_space_id: str = ""
    feishu_default_folder_token: str = ""
    memory_store_path: str = "./memory_store"
    memory_top_k: int = 5
    memory_compressor_model: str = "deepseek-v4-flash"
    memory_extractor_model: str = "deepseek-v4-flash"
    memory_enabled: bool = True
    amap_api_key: str = ""
    mcp_servers: dict = {}
    disabled_lanes: str = ""  # Comma-separated lane names to disable, e.g. "SIMPLE_TOOL,WORKFLOW"
    # Ticket 03: Chat v2 (UniHarness) 配置 —— 默认关闭，灰度期由 Gateway 分桶控制
    # agent_v2_enabled=False 时 v1 Agent/Gateway/Frontend/RAG 行为完全不变；
    # VERCEL + agent_v2_enabled=True 同时出现时启动检查失败（见 agent_v2.startup）
    agent_v2_enabled: bool = False
    # UniHarness 不可变 commit pin（stub 替代时记录等价上游 revision）
    agent_v2_uniharness_pin: str = "stub-local-757bf3d"
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
        self.rag_enable_clarification = os.getenv("RAG_ENABLE_CLARIFICATION", "false").lower() == "true"
        # 01 票据：graph runtime 开关
        self.rag_use_graph_runtime = os.getenv("RAG_USE_GRAPH_RUNTIME", "false").lower() == "true"
        # 04 票据：并行检索子问题上限
        self.rag_max_sub_questions = int(os.getenv("RAG_MAX_SUB_QUESTIONS", "3"))
        # 10 票据：存储后端 cutover 配置（12 票据：新增 "memory" 显式后端选项）
        backend = os.getenv("RAG_STORAGE_BACKEND", "pgvector").lower()
        self.rag_storage_backend = backend if backend in {"pgvector", "sqlite", "memory"} else "pgvector"
        self.rag_sqlite_db_path = os.getenv("RAG_SQLITE_DB_PATH", self._default_runtime_dir("data/rag.sqlite"))
        self.rag_cutover_marker_path = os.getenv("RAG_CUTOVER_MARKER_PATH", self._default_runtime_dir("data/cutover-marker.json"))
        self.rag_migration_progress_path = os.getenv("RAG_MIGRATION_PROGRESS_PATH", self._default_runtime_dir("data/migration-progress.json"))
        self.rag_baseline_performance_path = os.getenv("RAG_BASELINE_PERFORMANCE_PATH", self._default_runtime_dir("data/baseline-performance.json"))
        self.feishu_app_id = os.getenv("FEISHU_APP_ID", "")
        self.feishu_app_secret = os.getenv("FEISHU_APP_SECRET", "")
        self.feishu_default_space_id = os.getenv("FEISHU_DEFAULT_SPACE_ID", "")
        self.feishu_default_folder_token = os.getenv("FEISHU_DEFAULT_FOLDER_TOKEN", "")
        self.memory_store_path = os.getenv("MEMORY_STORE_PATH", self._default_runtime_dir("memory_store"))
        self.memory_top_k = int(os.getenv("MEMORY_TOP_K", "5"))
        self.memory_compressor_model = os.getenv("MEMORY_COMPRESSOR_MODEL", "deepseek-v4-flash")
        self.memory_extractor_model = os.getenv("MEMORY_EXTRACTOR_MODEL", "deepseek-v4-flash")
        self.memory_enabled = os.getenv("MEMORY_ENABLED", "true").lower() == "true"
        self.amap_api_key = os.getenv("AMAP_API_KEY", "")
        self.mcp_servers = self._json_env("MCP_SERVERS")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        self.disabled_lanes = os.getenv("DISABLED_LANES", "")
        # Ticket 03：Chat v2 配置加载
        self.agent_v2_enabled = os.getenv("AGENT_V2_ENABLED", "false").lower() == "true"
        self.agent_v2_uniharness_pin = os.getenv(
            "AGENT_V2_UNIHARNESS_PIN", "stub-local-757bf3d"
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


settings = Settings().load()
