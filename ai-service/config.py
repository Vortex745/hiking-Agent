import os
import json
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, load_dotenv, set_key


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
    rag_sqlite_db_path: str = "./data/rag.sqlite"  # SQLite 是唯一检索存储后端
    datasource_db_path: str = ""
    datasource_secret_key: str = ""
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
    # 模型推理透出：默认开启（有推理就发 ``thought`` 事件，没有就不发）。
    # 关闭后 kernel 不接收 on_thought，SSE 也不会出现该事件类型。
    agent_v2_thoughts_enabled: bool = True
    # Langfuse LLM 可观测性（可选）：未配置 key 时自动禁用，零回归。
    langfuse_enabled: bool = True
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"
    # OpenTelemetry 头部采样比例（ParentBased(TraceIDRatioBased)）。
    observability_sample_ratio: float = 1.0
    # stdout 日志配置：level 为 Python 日志级别，format 支持 json/text。
    log_level: str = "INFO"
    log_format: str = "json"
    # 本地 Agent 执行日志（LLM_LOG）：默认关闭，仅本地调试开启。
    llm_log_enabled: bool = False
    llm_log_path: str = "./logs/llm.log"

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
        # 01 票据：graph runtime 开关
        # 04 票据：并行检索子问题上限
        self.rag_sqlite_db_path = os.getenv("RAG_SQLITE_DB_PATH", self._default_runtime_dir("data/rag.sqlite"))
        self.datasource_db_path = os.getenv("DATASOURCE_DB_PATH", "") or str(
            Path(self.rag_sqlite_db_path).with_name("datasource.sqlite")
        )
        self.datasource_secret_key = os.getenv("DATASOURCE_SECRET_KEY", "")
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
        # 推理透出默认开启；运营可用 AGENT_V2_THOUGHTS_ENABLED=false 立即关闭。
        self.agent_v2_thoughts_enabled = (
            os.getenv("AGENT_V2_THOUGHTS_ENABLED", "true").lower() == "true"
        )
        # Langfuse：未配置 key 时 observability 模块返回空 callbacks，零回归。
        self.langfuse_enabled = os.getenv("LANGFUSE_ENABLED", "true").lower() == "true"
        self.langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        self.langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        self.langfuse_base_url = os.getenv(
            "LANGFUSE_BASE_URL", "https://cloud.langfuse.com"
        )
        try:
            self.observability_sample_ratio = float(
                os.getenv("OBSERVABILITY_SAMPLE_RATIO", "1.0")
            )
        except ValueError:
            self.observability_sample_ratio = 1.0
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.log_format = os.getenv("LOG_FORMAT", "json").lower()
        self.llm_log_enabled = (
            os.getenv("LLM_LOG_ENABLED", "false").lower() == "true"
        )
        self.llm_log_path = os.getenv(
            "LLM_LOG_PATH", self._default_runtime_dir("logs/llm.log")
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


# --- 运行时配置热更新 -------------------------------------------------------
# 本模块是唯一知道 .env 键名与文件格式的地方；路由层只提交字段字典。

_RUNTIME_CONFIG_LOCK = threading.Lock()

_SECTION_FIELD_ENV_KEYS: dict[str, dict[str, str]] = {
    "llm": {
        "base_url": "OPENAI_BASE_URL",
        "api_key": "OPENAI_API_KEY",
        "model": "OPENAI_MODEL",
    },
    "embedding": {
        "base_url": "EMBEDDING_BASE_URL",
        "api_key": "EMBEDDING_API_KEY",
        "model": "EMBEDDING_MODEL",
        "dimensions": "EMBEDDING_DIMENSIONS",
    },
    "rerank": {
        "base_url": "RERANK_BASE_URL",
        "api_key": "RERANK_API_KEY",
        "model": "RERANK_MODEL",
    },
}

_SECTION_FIELD_SETTINGS_ATTRS: dict[str, dict[str, str]] = {
    "llm": {
        "base_url": "openai_base_url",
        "api_key": "openai_api_key",
        "model": "openai_model",
    },
    "embedding": {
        "base_url": "embedding_base_url",
        "api_key": "embedding_api_key",
        "model": "embedding_model",
        "dimensions": "embedding_dimensions",
    },
    "rerank": {
        "base_url": "rerank_base_url",
        "api_key": "rerank_api_key",
        "model": "rerank_model",
    },
}


def _parse_runtime_dimensions(value: Any) -> int:
    """把 dimensions 解析为正整数；布尔/小数/非数字/非正数一律拒绝。"""
    if isinstance(value, bool):
        raise ValueError(f"dimensions must be a positive integer, got: {value!r}")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"dimensions must be a positive integer, got: {value!r}")
        value = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not re.fullmatch(r"\d+", text):
            raise ValueError(f"dimensions must be a positive integer, got: {value!r}")
        value = int(text)
    if not isinstance(value, int):
        raise ValueError(f"dimensions must be a positive integer, got: {value!r}")
    if value <= 0:
        raise ValueError(f"dimensions must be a positive integer, got: {value!r}")
    return value


def _parse_runtime_text(section: str, field: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{section}.{field} must be a non-empty string")
    return value.strip()


def _read_env_values(env_path: Path) -> dict[str, str]:
    """读取现有 .env 的原始值（不做变量插值），用于判断实际变更的键。"""
    if not env_path.is_file():
        return {}
    try:
        values = dotenv_values(env_path, interpolate=False)
    except (OSError, ValueError):
        return {}
    return {key: value for key, value in values.items() if value is not None}


def _set_env_key_with_retry(tmp_path: Path, key: str, value: str) -> None:
    """dotenv.set_key 的 Windows 兼容包装。

    Windows 上刚写出的临时文件可能被索引/杀毒进程短暂占用，set_key 内部的
    os.replace 会抛 PermissionError。set_key 失败时目标文件保持不变，重试安全。
    """
    delay = 0.02
    for attempt in range(6):
        try:
            set_key(str(tmp_path), key, value)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.2)


def _write_env_atomic(env_path: Path, updates: dict[str, str]) -> None:
    """同目录临时文件 + os.replace 原子替换，保留其他键与注释。

    临时文件与目标同目录，避免跨盘 os.replace 失败；替换前对副本调用
    dotenv.set_key，原文件在替换成功前保持不变。
    """
    env_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{env_path.name}.", suffix=".tmp", dir=str(env_path.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        if env_path.is_file():
            shutil.copy2(env_path, tmp_path)
        for key, value in updates.items():
            _set_env_key_with_retry(tmp_path, key, value)
        os.replace(tmp_path, env_path)
    except BaseException:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


def update_runtime_config(
    sections: dict[str, dict[str, Any]],
    env_path: str | Path | None = None,
) -> list[str]:
    """校验并持久化 LLM/embedding/rerank 配置，同时热更新进程内 settings。

    Args:
        sections: 形如 {"llm": {"base_url": ..., "model": ..., "api_key": ...}}；
            api_key 为 None 或空串表示保持原值（不写文件、不改内存）。
        env_path: 目标 .env；默认 ai-service/.env。

    Returns:
        实际发生变化的 env 键名（值未变化的键不计入）。

    Raises:
        ValueError: sections 结构、字段名或字段值非法（此时不落盘、不改内存）。
        OSError: .env 不可写（此时 settings/os.environ 保持原值）。
    """
    if not isinstance(sections, dict):
        raise ValueError("sections must be a mapping")

    unknown_sections = set(sections) - set(_SECTION_FIELD_ENV_KEYS)
    if unknown_sections:
        raise ValueError(f"unknown config sections: {sorted(unknown_sections)}")

    # 先完成全部解析与校验，校验失败不得留下半更新状态。
    env_updates: dict[str, str] = {}
    settings_updates: dict[str, Any] = {}
    for section_name, section in sections.items():
        if not isinstance(section, dict):
            raise ValueError(f"section {section_name!r} must be a mapping")
        field_env_keys = _SECTION_FIELD_ENV_KEYS[section_name]
        unknown_fields = set(section) - set(field_env_keys)
        if unknown_fields:
            raise ValueError(
                f"unknown fields in section {section_name!r}: {sorted(unknown_fields)}"
            )
        for field, env_key in field_env_keys.items():
            if field not in section:
                continue
            raw_value = section[field]
            if field == "api_key":
                if raw_value is None:
                    continue
                if not isinstance(raw_value, str):
                    raise ValueError(f"{section_name}.api_key must be a string")
                value = raw_value.strip()
                if not value:
                    continue
            elif field == "dimensions":
                value = _parse_runtime_dimensions(raw_value)
            else:
                value = _parse_runtime_text(section_name, field, raw_value)
            env_updates[env_key] = str(value)
            settings_updates[_SECTION_FIELD_SETTINGS_ATTRS[section_name][field]] = value

    if not env_updates:
        return []

    target = Path(env_path) if env_path is not None else ENV_PATH

    with _RUNTIME_CONFIG_LOCK:
        current_values = _read_env_values(target)
        changed = [key for key, value in env_updates.items() if current_values.get(key) != value]
        if changed:
            _write_env_atomic(target, {key: env_updates[key] for key in changed})
        # 持久化成功后再应用，写盘失败时内存状态保持不变。
        for attr, value in settings_updates.items():
            setattr(settings, attr, value)
        for key, value in env_updates.items():
            os.environ[key] = value

    return changed


ENV_PATH = Path(__file__).with_name(".env")


def env_file_updated_at() -> str | None:
    """返回 .env 最后修改时间（UTC ISO8601）；文件不存在为 None。"""
    from datetime import datetime, timezone

    try:
        mtime = ENV_PATH.stat().st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")
