"""受控评测环境：真实 FastAPI 路由 + 可注入替身（被跟踪的稳定评估包）。

设计（与 ai-service 既有测试同构，但面向独立评测进程）：

- 真实组件：chat_v2 / memory_v2 路由、RunRepository（临时 SQLite）、
  MemoryService（临时 SQLite）、artifacts 导出管线（真实字节落盘）。
- 替身：内核回合（agent_v2.kernel.run_chat_turn 属性补丁）、
  沙箱 Computer（chat_v2._build_artifact_computer 补丁）、
  领域委托（tools.hiking_domain.* / tools.hiking_knowledge.* 属性补丁）、
  权限判定（fixture.permission_gate，可选）。

每个 case 一个独立 FixtureEnv（独立临时目录 + 独立 SQLite + 独立主体），
互不共享状态；close() 恢复全部补丁，保证运行可复盘且不污染进程全局。
本模块只读临时目录，绝不触碰真实会话、配置或产物路径。
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .kernels import build_kernel

# ai-service 必须可导入（评测包从仓库根的 evals/ 目录运行）
_AI_SERVICE = Path(__file__).resolve().parents[2] / "ai-service"
if str(_AI_SERVICE) not in sys.path:
    sys.path.insert(0, str(_AI_SERVICE))


def _load_env() -> None:
    """加载 ai-service/.env（config.settings 导入时需要环境变量）。

    ``override=True``：从仓库根目录运行时，根 ``.env``（compose 用）可能已被
    先加载；评测必须使用 ai-service 的模型配置，不能被根配置覆盖。
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(_AI_SERVICE / ".env", override=True)
    except Exception:
        pass


_load_env()


class _PatchGuard:
    """setattr 补丁管理：记录 (obj, attr, original)，close 时恢复。"""

    def __init__(self) -> None:
        self._originals: list[tuple[Any, str, Any]] = []

    def set(self, obj: Any, attr: str, value: Any) -> None:
        self._originals.append((obj, attr, getattr(obj, attr, None)))
        setattr(obj, attr, value)

    def restore(self) -> None:
        for obj, attr, original in reversed(self._originals):
            try:
                setattr(obj, attr, original)
            except Exception:
                pass
        self._originals.clear()


class FakeArtifactComputer:
    """run 作用域沙箱 Computer 替身：内存文件 + 当前 product 契约。

    与当前 Computer 接口对齐（create_workspace / generate_pdf /
    build_manifest / read_artifact_bytes / run_bash / cleanup）：

    - ``generate_pdf`` 用产品自身渲染器产出**合法 PDF 字节**（对齐
      T05/T06 产物契约：文本伪 PDF 已不可行）；
    - ``run_bash`` 只返回脚本化 BashResult 并记录调用，**绝不执行真实
      宿主命令**（对齐 ProductSandboxBridge/run_bash 形状）。

    行为由 fixture.artifact_computer 配置：
      {"mode": "fake", "generate_pdf_error": false, "read_error": false,
       "cleanup_error": false, "bash_error": false,
       "bash_script": {"default": {"exit_code": 0, "stdout": "", "stderr": ""},
                       "prefix:echo ": {...}}}
    全部记录到 audit（创建/清理/导出/bash 尝试），供 scorer 验证生命周期。
    """

    sandbox_type = "local_native"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = dict(config or {})
        self.generate_pdf_error = bool(cfg.get("generate_pdf_error", False))
        self.read_error = bool(cfg.get("read_error", False))
        self.cleanup_error = bool(cfg.get("cleanup_error", False))
        self.bash_error = bool(cfg.get("bash_error", False))
        self._bash_script: dict[str, Any] = dict(cfg.get("bash_script") or {})
        self.files: dict[str, dict[str, bytes]] = {}  # workspace -> {path: bytes}
        self.created_workspaces: list[str] = []
        self.cleanup_calls: list[str] = []
        self.bash_calls: list[dict[str, Any]] = []

    # ---- Computer 协议 ----------------------------------------------------

    def create_workspace(self, run_id: str) -> str:
        workspace = f"/ws-eval/{run_id}"
        self.created_workspaces.append(workspace)
        self.files.setdefault(workspace, {})
        return workspace

    def cleanup(self, workspace_path: str) -> Any:
        self.cleanup_calls.append(workspace_path)
        removed = len(self.files.pop(workspace_path, {}))
        if self.cleanup_error:
            from agent_v2.computer import SandboxCleanupError

            raise SandboxCleanupError("eval: simulated cleanup failure")
        return SimpleNamespace(
            to_dict=lambda: {
                "duration_ms": 1,
                "files_written": removed,
                "bytes_written": 0,
            }
        )

    def generate_pdf(self, workspace_path: str, content: str, filename: str) -> str:
        """真实渲染器产出合法 PDF 字节；失败映射为当前产品错误类型。"""
        if self.generate_pdf_error:
            from agent_v2.computer import SandboxGenerationError

            raise SandboxGenerationError("eval: simulated pdf generation failure")
        from agent_v2.pdf_rendering import PDFRenderingError, render_markdown_pdf_bytes

        safe_name = Path(filename).name  # strip any path components（与产品一致）
        if not safe_name.endswith(".pdf"):
            safe_name = f"{safe_name}.pdf"
        try:
            data = render_markdown_pdf_bytes(content, title=Path(safe_name).stem)
        except PDFRenderingError as exc:
            from agent_v2.computer import SandboxGenerationError

            raise SandboxGenerationError(f"eval: pdf rendering failed: {exc}") from exc
        self.files[workspace_path][safe_name] = data
        return safe_name

    def build_manifest(
        self, workspace_path: str, relative_path: str, declared_mime: str
    ) -> Any:
        from agent_v2.artifacts import ArtifactManifest
        from agent_v2.computer import SandboxFileNotFoundError

        data = self.files.get(workspace_path, {}).get(relative_path)
        if data is None:
            raise SandboxFileNotFoundError(f"eval: no such file {relative_path!r}")
        return ArtifactManifest(
            relative_path=relative_path,
            public_filename=Path(relative_path).name,
            declared_mime=declared_mime,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )

    def read_artifact_bytes(self, workspace_path: str, relative_path: str) -> bytes:
        from agent_v2.computer import SandboxFileNotFoundError

        if self.read_error:
            raise SandboxFileNotFoundError("eval: simulated read failure")
        data = self.files.get(workspace_path, {}).get(relative_path)
        if data is None:
            raise SandboxFileNotFoundError(f"eval: no such file {relative_path!r}")
        return data

    def run_bash(
        self,
        workspace_path: str,
        command: str,
        *,
        timeout_seconds: float = 30.0,
        max_output_bytes: int = 1_048_576,
    ) -> Any:
        """脚本化 BashResult：记录调用，绝不执行真实宿主命令。"""
        self.bash_calls.append(
            {
                "workspace": workspace_path,
                "command": command,
                "timeout_seconds": timeout_seconds,
            }
        )
        if self.bash_error:
            from agent_v2.computer import ComputerError

            raise ComputerError("eval: simulated bash execution failure")
        from agent_v2.computer import BashResult

        scripted = self._scripted_bash(command)
        stdout = str(scripted.get("stdout", "") or "")
        stderr = str(scripted.get("stderr", "") or "")
        truncated = bool(scripted.get("truncated", False)) or (
            len(stdout.encode("utf-8")) > max_output_bytes
            or len(stderr.encode("utf-8")) > max_output_bytes
        )
        return BashResult(
            exit_code=int(scripted.get("exit_code", 0)),
            stdout=stdout[:max_output_bytes],
            stderr=stderr[:max_output_bytes],
            duration_ms=int(scripted.get("duration_ms", 1)),
            truncated=truncated,
        )

    def _scripted_bash(self, command: str) -> dict[str, Any]:
        for key, entry in self._bash_script.items():
            if key == "default":
                continue
            prefix = key[len("prefix:"):] if key.startswith("prefix:") else key
            if command.startswith(prefix):
                return dict(entry)
        return dict(self._bash_script.get("default") or {"exit_code": 0, "stdout": ""})


class FakeKBRetriever:
    """固定知识库的确定性检索：关键词重合度排序，平局按入库顺序。"""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = [dict(d) for d in docs]
        self.query_log: list[str] = []

    @staticmethod
    def _score(question: str, doc: dict[str, Any]) -> int:
        keywords = [w for w in str(doc.get("keywords", "")).split(",") if w]
        return sum(1 for k in keywords if k and k in question)

    def search(self, question: str, k: int = 4) -> list[dict[str, Any]]:
        self.query_log.append(question)
        scored = [
            (self._score(question, doc), idx, doc)
            for idx, doc in enumerate(self._docs)
        ]
        relevant = [x for x in scored if x[0] > 0]
        relevant.sort(key=lambda x: (-x[0], x[1]))  # 稳定排序：平局保入库序
        hits = [dict(doc) for _, _, doc in relevant[:k]]
        return hits


class GroundedAugmenter:
    """确定性 augmenter：仅基于命中文档合成结论 + 来源引用。

    rag.graph.aggregate_answer 以 LangChain Document 调用 augmenter
    （page_content + metadata.source/title）；本类同时兼容 dict 形态。
    无命中（或检索为空）时由调用方（rag.graph）给出"未找到相关文档"。
    有命中时输出：结论行 + 逐条来源引用。
    """

    @staticmethod
    def _doc_fields(doc: Any) -> tuple[str, str, str]:
        if isinstance(doc, dict):
            content = str(doc.get("content") or doc.get("conclusion") or "")
            source = str(doc.get("source") or "")
            title = str(doc.get("title") or "")
        else:  # LangChain Document
            content = str(getattr(doc, "page_content", "") or "")
            meta = getattr(doc, "metadata", None) or {}
            source = str(meta.get("source") or "")
            title = str(meta.get("title") or "")
        return content.strip(), source.strip(), title.strip()

    def augment(self, question: str, docs: list[Any]) -> str:
        lines: list[str] = []
        for doc in docs or []:
            content, source, title = self._doc_fields(doc)
            if content:
                lines.append(f"- {content}（来源：{source or title}）")
        if not lines:
            return "知识库中没有足够的证据回答该问题。"
        return "根据知识库证据：\n" + "\n".join(lines)


class FixtureEnv:
    """一个评测用例的受控环境。"""

    def __init__(self, fixture: dict[str, Any], env_dir: Path, live: bool = False) -> None:
        self.fixture = fixture
        self.env_dir = env_dir
        self.live = live
        env_dir.mkdir(parents=True, exist_ok=True)
        self.patch = _PatchGuard()
        self.variables: dict[str, str] = {}
        self.computer: FakeArtifactComputer | None = None
        self.rag_retriever: FakeKBRetriever | None = None
        self.kernel: Any = None
        self._stores: list[Any] = []

        self._init_settings()
        self._init_runtime_stores()
        self._init_subjects()
        self._init_memories()
        self._init_kernel()
        self._init_delegate_overrides()
        self._init_permission_gate()
        self._init_artifact_computer()
        self._init_rag()

    # ---- 初始化 -----------------------------------------------------------

    def _init_settings(self) -> None:
        from config import settings

        self.patch.set(settings, "agent_v2_enabled", True)
        self.patch.set(settings, "agent_v2_state_dir", str(self.env_dir))
        self.patch.set(settings, "agent_v2_sandbox_type", "local_native")
        # 相对 RAG 路径按 ai-service 解析，避免从仓库根目录运行时指向
        # 不存在的 ./data/rag.sqlite 而退化成"无证据"路径。
        rag_path = Path(getattr(settings, "rag_sqlite_db_path", "") or "")
        if rag_path and not rag_path.is_absolute():
            self.patch.set(
                settings,
                "rag_sqlite_db_path",
                str((_AI_SERVICE / rag_path).resolve()),
            )
        # 编排层在内核回合前检查 API Key（run_orchestration 决策 #27）；
        # 评测用替身内核，这里放行该守卫。live 模式需要真实凭据，不放行。
        if not self.live and not getattr(settings, "openai_api_key", ""):
            self.patch.set(settings, "openai_api_key", "sk-eval-placeholder")

    def _init_runtime_stores(self) -> None:
        from agent_v2.persistence import CheckpointsStore, ControlStore
        from agent_v2.repositories.memory import MemoryStore
        from agent_v2.memory_service import MemoryService
        from agent_v2.runs import RunRepository

        self.runs_repo = RunRepository(
            control=ControlStore(str(self.env_dir / "control.sqlite")),
            checkpoints=CheckpointsStore(str(self.env_dir / "checkpoints.sqlite")),
        )
        self.memory_service = MemoryService(
            MemoryStore(str(self.env_dir / "memory.sqlite"))
        )
        self._stores.append(self.runs_repo)
        self._stores.append(self.memory_service)

    def _init_subjects(self) -> None:
        from agent_v2.memory_service import generate_subject_token

        for name in self.fixture.get("subjects") or []:
            self.variables[str(name)] = generate_subject_token()

    def _init_memories(self) -> None:
        """按 fixture.memories 为各主体播种偏好记忆（隔离评测的数据底座）。

        {"alice": ["偏好A", {"content": "偏好B", "ttl_days": -1}]}
        dict 形态支持 ttl_days（负数 = 已过期，用于过期降级评测）。
        save_preference 要求主体先开启 memory_enabled，播种后恢复原开关。
        """
        memories = self.fixture.get("memories") or {}
        if not memories:
            return
        for name, contents in memories.items():
            token = self.variables.get(str(name))
            if not token:
                continue
            subject_id = self.memory_service.resolve_subject(token).subject_id
            was_enabled = self.memory_service.memory_enabled(subject_id)
            self.memory_service.set_memory_enabled(subject_id, True)
            for content in contents or []:
                if isinstance(content, dict):
                    text = str(content.get("content", ""))
                    ttl_days = content.get("ttl_days")
                else:
                    text, ttl_days = str(content), None
                memory_id = self.memory_service.save_preference(subject_id, text)
                if memory_id and ttl_days is not None:
                    self.memory_service.update_memory(
                        subject_id,
                        memory_id,
                        content=text,
                        category="preference",
                        ttl_days=int(ttl_days),
                    )
            if not was_enabled:
                self.memory_service.set_memory_enabled(subject_id, False)

    def _init_kernel(self) -> None:
        # live 模式不替换内核（保留真实 LLM 回合，需真实凭据）
        if self.live:
            return
        self.kernel = build_kernel(self.fixture)
        if self.kernel is None:
            return
        from agent_v2 import kernel as kernel_module

        self.patch.set(kernel_module, "run_chat_turn", self.kernel)

    def _init_delegate_overrides(self) -> None:
        """按 fixture.delegate_overrides 注入工具委托的失败模式。"""
        overrides = self.fixture.get("delegate_overrides") or []
        if not overrides:
            return

        import tools.hiking_domain as domain_module
        import tools.hiking_knowledge as knowledge_module

        for item in overrides:
            tool_name = str(item["tool"])
            behavior = str(item.get("behavior", "ok"))
            # 委托按工具名解析所在模块（kernel_tools._delegate_for 同源）
            module = (
                knowledge_module
                if tool_name == "hiking_knowledge_search"
                else domain_module
            )
            original = getattr(module, tool_name)
            delay = float(item.get("delay_seconds", 31.0))

            if behavior == "timeout":

                async def _timeout(*args: Any, _delay: float = delay, **kwargs: Any):
                    await asyncio.sleep(_delay)
                    return None

                self.patch.set(module, tool_name, _timeout)
            elif behavior == "error":

                async def _error(*args: Any, **kwargs: Any):
                    raise RuntimeError(f"eval: simulated {tool_name} failure")

                self.patch.set(module, tool_name, _error)
            elif behavior == "incomplete":
                data = dict(item.get("data") or {})

                async def _incomplete(*args: Any, _d: dict = data, **kwargs: Any):
                    return _d

                self.patch.set(module, tool_name, _incomplete)
            # behavior == "ok"：保留 original（显式恢复以防外层已污染）
            else:
                self.patch.set(module, tool_name, original)

    def _init_permission_gate(self) -> None:
        """按 fixture.permission_gate 注入 PermissionGate 判定。

        "allow" —— 所有工具判定 ALLOW（跳过 approval 暂停，驱动完整
        artifact 流）；"deny" —— 全部 DENY（tool_denied 终止路径）。
        不配置时保留产品真实 gate（generate_pdf → NEEDS_APPROVAL）。
        """
        mode = self.fixture.get("permission_gate")
        if mode not in ("allow", "deny"):
            return
        import api.chat_v2 as chat_v2_module
        from agent_v2 import permissions as perms

        if mode == "allow":
            decision, risk = perms.PermissionDecision.ALLOW, perms.RiskLevel.LOW
        else:
            decision, risk = perms.PermissionDecision.DENY, perms.RiskLevel.HIGH

        class _FixedGate:
            def evaluate(self, profile: Any, tool_name: str, args: Any = None) -> Any:
                return perms.PermissionVerdict(
                    decision=decision, risk=risk, reason=f"eval: {mode}"
                )

        self.patch.set(chat_v2_module, "PermissionGate", _FixedGate)

    def _init_artifact_computer(self) -> None:
        cfg = self.fixture.get("artifact_computer")
        if cfg is None:  # 空字典 {} 是合法配置（默认行为），不能当作未配置
            return
        self.computer = FakeArtifactComputer(cfg)
        import api.chat_v2 as chat_v2_module

        self.patch.set(
            chat_v2_module,
            "_build_artifact_computer",
            lambda *a, **k: self.computer,
        )
        self.variables.setdefault("artifact_computer", "fake")

    def _init_rag(self) -> None:
        # Spec #13 T05/T06: /api/v1/rag/query runs as a kernel run
        # (profile=rag); the fixture no longer injects a RAGRuntime.
        # The "rag" fixture key only mounts the router; document
        # evidence comes from the scripted kernel audits.
        self.rag_retriever = None
        self.rag_runtime = None

    # ---- 应用 ------------------------------------------------------------

    def build_app(self) -> Any:
        from fastapi import FastAPI

        import api.chat_v2 as chat_v2_module
        import api.memory_v2 as memory_v2_module

        # A11: the HTTP RAG surface is gone — retrieval rides the chat
        # knowledge tool, so rag_grounding cases drive /api/v2/chat/sse.
        app = FastAPI()
        app.include_router(chat_v2_module.chat_v2_router, prefix="/api/v2")
        app.include_router(chat_v2_module.artifacts_v2_router, prefix="/api/v2")
        app.include_router(chat_v2_module.approvals_v2_router, prefix="/api/v2")
        app.include_router(memory_v2_module.memory_v2_router, prefix="/api/v2")

        class _EvalRuntime:
            runs = self.runs_repo
            memory = self.memory_service
            telemetry = None
            capacity = None
            kernel_checkpoints = None

        app.state.agent_v2_runtime = _EvalRuntime()
        if self.fixture.get("rag"):
            app.state.rag_runtime = self.rag_runtime
        return app

    # ---- 资源释放 ---------------------------------------------------------

    def audit(self) -> dict[str, Any]:
        """供报告记录的环境元数据（不含敏感内容）。"""
        info: dict[str, Any] = {
            "env_dir": self.env_dir.name,
            "kernel": type(self.kernel).__name__ if self.kernel else None,
        }
        if self.computer is not None:
            info["artifact_computer"] = {
                "created_workspaces": len(self.computer.created_workspaces),
                "created_workspace_paths": list(self.computer.created_workspaces),
                "cleanup_calls": len(self.computer.cleanup_calls),
                "cleanup_workspace_paths": list(self.computer.cleanup_calls),
                "files_remaining": sum(
                    len(files) for files in self.computer.files.values()
                ),
            }
        if self.rag_retriever is not None:
            info["rag_queries"] = list(self.rag_retriever.query_log)
        return info

    def close(self) -> None:
        self.patch.restore()
        for store in self._stores:
            closer = getattr(store, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass
        self._stores.clear()
