"""The single Agent v2 kernel assembly (Ticket 26).

``assemble_kernel_agent`` is the ONLY place in the product that constructs a
real UniHarness agent. Request handlers and profile flows call it with a
fully-specified :class:`KernelAssemblyConfig`; they never import
``uniharness`` themselves, never pick models or Computers, and cannot
assemble a competing runtime.

Fail-closed rules:

* The loaded UniHarness must match the configured pin (identity re-checked
  here, independent of startup) — mismatch raises :class:`KernelAssemblyError`.
* Any model / Computer / assembly failure raises :class:`KernelRunError`
  carrying a stable public ``code``; callers map it to a terminal SSE
  ``error`` event and MUST NOT fall back to stubs, direct flows, or v1.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from observability.langfuse_setup import get_langchain_callbacks, langchain_run_config

logger = logging.getLogger("ai-service.agent_v2.kernel")

# Some providers (MiniMax-M3 via OpenAI-compatible endpoints) leak reasoning
# markers ("<think>…</think>", stray "</mm:think>") into assistant content.
# Public output must never carry them.
_REASONING_BLOCK = re.compile(r"<(?:think|thinking|reasoning)>.*?</(?:think|thinking|reasoning)>", re.DOTALL | re.IGNORECASE)
_REASONING_TAG = re.compile(r"</?(?:mm:)?(?:think|thinking|reasoning)>", re.IGNORECASE)


def _strip_reasoning(text: str) -> str:
    cleaned = _REASONING_BLOCK.sub("", text)
    return _REASONING_TAG.sub("", cleaned)

# Built-in tool groups exposed by the vendored fork's create_agent.
BUILTIN_TOOL_GROUPS: frozenset[str] = frozenset({"cli", "web", "todo", "task", "skills"})


class KernelAssemblyError(RuntimeError):
    """Kernel could not be assembled from the provided configuration."""


class KernelModelFailure(RuntimeError):
    """Stable terminal failure raised for model/dependency errors.

    ``code`` is a public, user-safe error code; ``detail`` must never
    include credentials, sandbox paths, prompts, graph state, or raw
    exception text.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class KernelAssemblyConfig:
    """Everything the unique assembly needs; nothing more may be injected.

    * ``profile_id`` / ``builtin_tool_groups`` — the server-resolved profile
      surface. An empty group set yields a pure LLM loop (no shell tools).
    * ``system_prompt`` — product-composed prompt built AFTER request
      normalization; the request layer may normalize inputs but must not
      plan tools or execute agent steps.
    * ``compaction_threshold`` — server-owned token budget for short-term
      context compaction.
    * ``computer`` — a run-scoped object satisfying the upstream Computer
      protocol (already provisioned by the product computer factory or a
      deterministic test double). The kernel owns its lifetime inside the
      returned agent handle.
    * ``checkpointer`` — optional LangGraph checkpointer owned by the
      caller's persistence layer.
    """

    profile_id: str
    system_prompt: str
    computer: Any
    builtin_tool_groups: tuple[str, ...] = ()
    compaction_threshold: int | None = None
    output_limit: int | None = None
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    subagent_definitions: dict[str, Any] = field(default_factory=dict)
    checkpointer: Any | None = None
    # Toolless deployments (pure chat loop) skip the Computer environment
    # probe entirely so the sandbox is never touched at assembly time.
    resolve_environment: bool = True

    # Ticket 28: bounded domain tool adapters and the pre-configured kernel
    # permission gate (product bridge rule attached). Both are constructed
    # per-turn by the product layer and forwarded verbatim.
    extra_tools: tuple[Any, ...] = ()
    permission_gate: Any | None = None

    # Ticket 31: durable-approval guard invoked at the kernel tool-call
    # boundary (NEEDS_APPROVAL branch). Signature: async (tool_name, args,
    # prompt) -> bool. Must be effect-ledger backed — see CoworkBashTool.
    approval_callback: Any | None = None

    def __post_init__(self) -> None:
        unknown = set(self.builtin_tool_groups) - BUILTIN_TOOL_GROUPS
        if unknown:
            raise KernelAssemblyError(
                f"Unknown builtin tool groups requested: {sorted(unknown)}"
            )


def build_chat_model(settings: Any) -> Any:
    """Build the OpenAI-compatible chat model from server settings.

    Importing ChatOpenAI lazily keeps disabled deployments free of the
    dependency cost. Raises :class:`KernelModelFailure` with a stable code
    when credentials are absent so callers fail closed.
    """
    if not getattr(settings, "openai_api_key", ""):
        raise KernelModelFailure(
            "model_unavailable",
            "主模型未配置 API Key (OPENAI_API_KEY 为空)。",
        )
    try:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            temperature=0,
            max_retries=3,
        )
    except Exception as exc:
        raise KernelModelFailure(
            "kernel_model_build_failed",
            "主模型初始化失败。",
        ) from exc


def _verify_pin(settings: Any) -> None:
    """Re-verify kernel identity at assembly time (defense in depth)."""
    from agent_v2.uniharness.kernel_identity import resolve_kernel_identity

    pin = getattr(settings, "agent_v2_uniharness_pin", "unknown")
    identity = resolve_kernel_identity(pin)
    if not identity.compatible:
        raise KernelAssemblyError(identity.detail)
    logger.info(
        "kernel identity verified at assembly (fork_commit=%s)",
        identity.fork_commit,
    )


async def assemble_kernel_agent(config: KernelAssemblyConfig, settings: Any) -> Any:
    """Assemble ONE real UniHarness agent. The product's only construction site.

    Returns an open :class:`uniharness.langchain.agent.Agent`. Callers own
    closing it (``await agent.aclose()`` / async-with) exactly once, in a
    finally block.
    """
    _verify_pin(settings)

    # Sole import site of the real kernel factory in production code.
    from uniharness.langchain.agent import create_agent as kernel_create_agent

    chat_model = build_chat_model(settings)
    model_input: Any = chat_model
    if config.compaction_threshold is not None:
        from uniharness.harness.model import ModelProfile as KernelModelProfile

        model_input = KernelModelProfile(
            model=chat_model,
            compaction_threshold=config.compaction_threshold,
        )

    try:
        agent = await kernel_create_agent(
            model_input,
            config.computer,
            skill_paths=(),
            system_prompt=config.system_prompt,
            checkpointer=config.checkpointer,
            agents=dict(config.subagent_definitions),
            builtin_tool_groups=(
                tuple(config.builtin_tool_groups)
                if config.builtin_tool_groups
                else ()
            ),
            resolve_environment=config.resolve_environment,
            extra_tools=list(config.extra_tools),
            **({"permission_gate": config.permission_gate} if config.permission_gate is not None else {}),
            **({"approval_callback": config.approval_callback} if config.approval_callback is not None else {}),
            **({"mcp_servers": dict(config.mcp_servers)} if config.mcp_servers else {}),
        )
    except KernelModelFailure:
        raise
    except Exception as exc:
        raise KernelAssemblyError("内核装配失败，请稍后重试。") from exc
    return agent


class NullKernelComputer:
    """Upstream-Computer-protocol stand-in for toolless kernel runs.

    Registered so ``create_agent``'s required parameter is satisfied without
    provisioning any sandbox. With ``builtin_tool_groups=()`` and
    ``resolve_environment=False`` nothing ever invokes it; a stray invocation
    (a bug) fails closed instead of silently touching the host.
    """

    workspace_hint = "/workspace"

    @property
    def is_running(self) -> bool:
        return False

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def run(self, command: str, *, timeout: float | None = None) -> Any:
        from uniharness.exceptions import CLIError

        raise CLIError("toolless kernel run attempted to execute a shell command")

    async def upload(self, src: str, dst: str) -> None:
        from uniharness.exceptions import CLIError

        raise CLIError("toolless kernel run cannot upload files")

    async def download(self, src: str, dst: str) -> None:
        from uniharness.exceptions import CLIError

        raise CLIError("toolless kernel run cannot download files")


def is_provider_throttled(exc: BaseException | None) -> bool:
    """True when the failure chain contains a provider rate-limit response."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        text = f"{type(exc).__name__}:{exc}".lower()
        if "ratelimit" in text or ("429" in text and "limit" in text):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def extract_final_text(agent_result: dict[str, Any]) -> str | None:
    """Map the finished kernel state to the public final assistant text.

    Returns ``None`` when the kernel produced no final AI content — callers
    convert that into their stable failure semantics rather than emitting an
    empty success.
    """
    messages = (agent_result or {}).get("messages") or []
    last_content: Any = None
    for message in reversed(messages):
        msg_type = getattr(message, "type", "")
        if msg_type == "ai":
            last_content = getattr(message, "content", None)
            break
    if isinstance(last_content, list):  # content blocks → joined text parts
        parts = []
        for block in last_content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        last_content = "".join(parts)
    text = _strip_reasoning(str(last_content)).strip() if last_content else ""
    return text or None


def build_kernel_messages(
    *,
    message: str,
    history: list[dict[str, str]] | None = None,
) -> list[tuple[str, str]]:
    """Ticket 05 (T08): single construction of the model input messages.

    The bounded server-side history (T07) is prepended in its given order,
    then the CURRENT user message is appended exactly once. Every kernel
    invocation style (``ainvoke`` for the final result today; an
    ``astream`` loop if the harness ever streams tokens) derives its input
    from this one function, so both paths observe the same history order
    and the same single current message.

    Defensive filter: only ``user`` / ``assistant`` entries with string
    content are forwarded — an internal tool/approval record can never
    leak into the model input even if a caller passes a mis-shaped list.
    History rows help the model understand the user's constraints; they
    grant no factual authority for the current turn (the evidence gate
    audits only THIS turn's tool results).
    """
    messages: list[tuple[str, str]] = []
    for entry in history or ():
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        content = entry.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content:
            messages.append((role, content))
    messages.append(("user", message))
    return messages


async def run_chat_turn(
    *,
    system_prompt: str,
    message: str,
    settings: Any,
    run_id: str | None = None,
    checkpoint_store: Any | None = None,
    extra_tools: tuple[Any, ...] = (),
    kernel_gate: Any | None = None,
    mcp_servers: dict[str, Any] | None = None,
    computer: Any | None = None,
    approval_callback: Any | None = None,
    subagent_definitions: dict[str, Any] | None = None,
    history: list[dict[str, str]] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Run ONE chat turn through the real UniHarness model loop (T28 entry).

    Returns ``(final_text, audit_entries)``. Audit records bind every tool
    execution and every permission denial to the public run identity so the
    stream layer can emit evidence events; adapters and the gate share one
    in-memory sink for the turn.

    ``history`` is the server-read bounded recent history (Ticket 05 /
    T07): it is prepended verbatim before the current user message, which
    is appended exactly once (:func:`build_kernel_messages`). The run's
    checkpoint thread stays ``v2:chat:{run_id}`` — history never changes
    the checkpoint identity.

    Failure semantics identical to Ticket 26: stable :class:`KernelModelFailure`
    codes, no fallback of any kind.
    """
    import asyncio

    checkpointer = None
    if checkpoint_store is not None:
        if not run_id:
            raise KernelModelFailure(
                "checkpoint_unavailable",
                "运行检查点需要公共 run identity，内部状态不一致。",
            )
        from agent_v2.kernel_persistence import KernelCheckpointUnavailable

        try:
            checkpointer = await checkpoint_store.get()
        except KernelCheckpointUnavailable as exc:
            raise KernelModelFailure(exc.code, exc.message) from exc

    audit_entries: list[dict[str, Any]] = []

    def _sink(entry: dict[str, Any]) -> None:
        audit_entries.append(entry)

    config = KernelAssemblyConfig(
        profile_id="chat",
        system_prompt=system_prompt,
        computer=computer if computer is not None else NullKernelComputer(),
        builtin_tool_groups=("task",) if subagent_definitions else (),
        resolve_environment=False,
        checkpointer=checkpointer,
        extra_tools=tuple(extra_tools),
        permission_gate=kernel_gate,
        mcp_servers=dict(mcp_servers) if mcp_servers else {},
        approval_callback=approval_callback,
        subagent_definitions=dict(subagent_definitions) if subagent_definitions else {},
    )
    try:
        agent = await assemble_kernel_agent(config, settings)
    except KernelAssemblyError as exc:
        logger.exception("chat kernel assembly failed")
        raise KernelModelFailure("kernel_unavailable", "Agent 内核暂不可用，请稍后重试。") from exc

    invoke_config = (
        {"configurable": {"thread_id": f"v2:chat:{run_id}"}}
        if checkpointer is not None
        else None
    )
    # Langfuse observability: callbacks propagate through the whole LangGraph
    # run tree (chain/tool/generation), session id rides on config metadata
    # via the request-bound ContextVar. Empty list when tracing is disabled.
    _lf_callbacks = get_langchain_callbacks()
    if _lf_callbacks:
        invoke_config = dict(invoke_config) if invoke_config else {}
        invoke_config["callbacks"] = _lf_callbacks
        invoke_config.update(langchain_run_config())
    # Ticket 05 (T08): ONE input construction for every kernel invocation —
    # bounded history first, the current user message appended exactly once.
    input_messages = build_kernel_messages(message=message, history=history)
    try:
        try:
            result = await asyncio.wait_for(
                agent.ainvoke({"messages": input_messages}, config=invoke_config),
                timeout=240,
            )
        except TimeoutError as exc:
            raise KernelModelFailure("model_timeout", "模型响应超时，请稍后重试。") from exc
        text = extract_final_text(result)
    except KernelModelFailure:
        raise
    except Exception as exc:
        logger.exception("chat kernel run failed")
        raise KernelModelFailure(
            "kernel_run_failed",
            "生成回复失败，请稍后重试。",
        ) from exc
    finally:
        await agent.aclose()

    if not text:
        raise KernelModelFailure("empty_model_output", "未生成有效回复，请重试。")
    return text, audit_entries


async def run_toolless_chat(
    *,
    system_prompt: str,
    message: str,
    settings: Any,
    run_id: str | None = None,
    checkpoint_store: Any | None = None,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Ticket-26-compatible wrapper: text-only result, no tools, no gate.

    Ticket 05: accepts the same ``history`` input and forwards it to the
    single :func:`run_chat_turn` construction, so both kernel entry points
    observe the identical history order and single current message.
    """
    text, _audit = await run_chat_turn(
        system_prompt=system_prompt,
        message=message,
        settings=settings,
        run_id=run_id,
        checkpoint_store=checkpoint_store,
        history=history,
    )
    return text
