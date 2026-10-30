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
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from observability import llm_log, tracing
from observability.callbacks import (
    ToolActivityCallbackHandler,
    ToolLogCallbackHandler,
    ToolSpanCallbackHandler,
)

logger = logging.getLogger("ai-service.agent_v2.kernel")

# Some providers (MiniMax-M3 via OpenAI-compatible endpoints) leak reasoning
# markers ("<think>…</think>", stray "</mm:think>") into assistant content.
# Public output must never carry them.
_REASONING_BLOCK = re.compile(r"<(?:mm:)?(?:think|thinking|reasoning)>.*?(?:</(?:mm:)?(?:think|thinking|reasoning)>|$)", re.DOTALL | re.IGNORECASE)
_REASONING_TAG = re.compile(r"</?(?:mm:)?(?:think|thinking|reasoning)>", re.IGNORECASE)


def _strip_reasoning(text: str) -> str:
    cleaned = _REASONING_BLOCK.sub("", text)
    return _REASONING_TAG.sub("", cleaned)


class _StreamingPublicText:
    """Strip reasoning even when a provider splits its tags across tokens.

    Hidden reasoning is not discarded: it accumulates in ``reasoning`` and
    each :meth:`feed` exposes the newly hidden text through
    :meth:`take_reasoning`, so the caller can publish it as a separate
    ``thought`` event while the public answer stays clean.
    """

    def __init__(self) -> None:
        self.pending = ""
        self.hidden = False
        self.started = False
        self.whitespace = ""
        self.reasoning = ""
        self._reasoning_delta = ""

    def feed(self, content: str, *, final: bool = False) -> str:
        self.pending += content
        visible: list[str] = []
        self._reasoning_delta = ""
        while self.pending:
            opening = self.pending.find("<")
            if opening == -1:
                self._collect(self.pending, visible)
                self.pending = ""
                break
            self._collect(self.pending[:opening], visible)
            self.pending = self.pending[opening:]
            closing = self.pending.find(">")
            if closing == -1:
                # Hold only a possible reasoning marker, not arbitrary '<' text.
                candidates = ("<think>", "</think>", "<thinking>", "</thinking>",
                              "<reasoning>", "</reasoning>", "<mm:think>", "</mm:think>")
                if not final and any(tag.startswith(self.pending.lower()) for tag in candidates):
                    break
                self._collect(self.pending, visible)
                self.pending = ""
                break
            tag = self.pending[:closing + 1]
            self.pending = self.pending[closing + 1:]
            if _REASONING_TAG.fullmatch(tag):
                self.hidden = not tag.startswith("</")
            else:
                self._collect(tag, visible)
        joined = self.whitespace + "".join(visible)
        if not self.started:
            joined = joined.lstrip()
        text = joined.rstrip()
        self.whitespace = joined[len(text):]
        if text:
            self.started = True
        return text

    def _collect(self, text: str, visible: list[str]) -> None:
        if not text:
            return
        if self.hidden:
            self.reasoning += text
            self._reasoning_delta += text
        else:
            visible.append(text)

    def take_reasoning(self) -> str:
        delta = self._reasoning_delta
        self._reasoning_delta = ""
        return delta


class _ThoughtBuffer:
    """Batch hidden reasoning into readable chunks with a per-turn cap."""

    def __init__(
        self,
        on_thought: Callable[[str], Awaitable[None]] | None,
        *,
        flush_chars: int = 180,
        max_chars: int = 8000,
    ) -> None:
        self._on_thought = on_thought
        self._buffer = ""
        self._emitted = 0
        self._flush_chars = flush_chars
        self._max_chars = max_chars

    async def add(self, text: str) -> None:
        if self._on_thought is None or not text:
            return
        self._buffer += text
        if len(self._buffer) >= self._flush_chars:
            await self.flush()

    async def flush(self) -> None:
        chunk, self._buffer = self._buffer, ""
        if self._on_thought is None or not chunk.strip():
            return
        remaining = self._max_chars - self._emitted
        if remaining <= 0:
            return
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
        self._emitted += len(chunk)
        await self._on_thought(chunk)


class _ModelRounds:
    """Track MAIN-model rounds and report each started/finished pair.

    Agent process tracking ticket 03: a "round" is one main-model step of
    the graph. Subagents, compaction and tool models never reach this
    tracker because the caller filters them out first. Exactly one
    ``started`` and one ``finished`` event leave per round, including the
    failure paths (the caller closes in a ``finally``).
    """

    def __init__(self, on_activity: Callable[[str, dict[str, Any]], None] | None) -> None:
        self._on_activity = on_activity
        self._step: Any = _ROUND_STEP_UNSET
        self._active = False
        self._started_at = 0.0
        self._index = 0

    def _push(self, payload: dict[str, Any]) -> None:
        if self._on_activity is None:
            return
        try:
            self._on_activity("model_round", payload)
        except Exception:  # pragma: no cover - reporting must never break a turn
            logger.exception("model round sink failure (non-fatal)")

    def observe(self, step: Any) -> None:
        """Open a round on a new main-model step; close the previous one."""
        if step == self._step:
            # Same step: never reopen after the round was closed by its own
            # tool decision.
            return
        self.close()
        self._step = step
        self._active = True
        self._index += 1
        self._started_at = time.perf_counter()
        self._push({"index": self._index, "step": step, "state": "started"})

    def close(self, state: str = "finished") -> None:
        if not self._active:
            return
        self._active = False
        self._push({
            "index": self._index,
            "step": self._step,
            "state": state,
            "duration_ms": round((time.perf_counter() - self._started_at) * 1000, 1),
        })

# Built-in tool groups exposed by the vendored fork's create_agent.
BUILTIN_TOOL_GROUPS: frozenset[str] = frozenset({"cli", "web", "todo", "task", "skills"})

_ROUND_STEP_UNSET = object()


class KernelAssemblyError(RuntimeError):
    """Kernel could not be assembled from the provided configuration."""


class KernelApprovalPending(RuntimeError):
    """The checkpointed graph stopped before an approval-gated tool call."""

    def __init__(self, details: dict[str, Any]) -> None:
        super().__init__("approval pending")
        self.details = details


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


def _map_invalid_tool_arguments(exc: BaseException) -> KernelModelFailure | None:
    """Translate the kernel's typed argument error into the public failure.

    The middleware owns detection and the one repair attempt; the kernel
    only maps the type. Codes, counts, and call ids are logged — never raw
    arguments, prompts, tool payloads, or the original exception text.
    """
    from uniharness.langchain.middleware import InvalidToolArgumentsError

    if not isinstance(exc, InvalidToolArgumentsError):
        return None
    logger.warning(
        "chat kernel invalid tool arguments: code=%s calls=%d ids=%s",
        exc.code,
        exc.call_count,
        ",".join(exc.call_ids) or "-",
    )
    return KernelModelFailure(
        "invalid_tool_arguments",
        "工具参数格式错误，本次请求未完成。",
    )


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

        from observability.generation import trace_chat_model

        return trace_chat_model(
            ChatOpenAI(
                model=settings.openai_model,
                base_url=settings.openai_base_url,
                api_key=settings.openai_api_key,
                temperature=0,
                max_retries=3,
            )
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
    invocation style (``ainvoke`` for the final result or ``astream``
    for incremental replies) derives its input
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


def _callbacks_for_turn(turn_span: Any, on_activity: Any = None) -> list[Any]:
    """Tool callbacks for this turn: spans and/or the local execution log.

    The log handler is independent of tracing so enabling LLM_LOG never
    requires Langfuse, and enabling tracing never duplicates log records.
    ``on_activity`` (agent process tracking ticket 02) adds the live
    activity channel; it is opt-in so no other kernel consumer pays for it.
    """
    handlers: list[Any] = []
    if on_activity is not None:
        handlers.append(ToolActivityCallbackHandler(on_activity))
    if tracing.tracing_enabled():
        handlers.append(ToolSpanCallbackHandler(parent_span=turn_span))
    if llm_log.enabled():
        handlers.append(ToolLogCallbackHandler())
    return handlers


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
    on_text: Callable[[str, list[dict[str, Any]]], Awaitable[None]] | None = None,
    on_thought: Callable[[str], Awaitable[None]] | None = None,
    on_activity: Callable[[str, dict[str, Any]], None] | None = None,
    history: list[dict[str, str]] | None = None,
    compaction_threshold: int | None = None,
    resume: bool = False,
    resume_interrupt_id: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Run ONE chat turn through the real UniHarness model loop (T28 entry).

    ``on_text`` receives filtered model deltas when supplied; callers enforce
    the evidence gate before publishing them. ``on_thought`` receives the
    reasoning the public text filter hid (``<think>`` blocks or a provider's
    ``reasoning_content`` delta); it is an opt-in side channel and never
    changes the public answer. Other kernel consumers retain the
    single-result invocation. Returns ``(final_text, audit_entries)``.
    Audit records bind every tool
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
        compaction_threshold=compaction_threshold,
    )
    with tracing.start_span(
        "agent.turn",
        {
            "run.id": run_id or "",
            "gen_ai.operation.name": "invoke_agent",
        },
    ) as turn_span:
        # Started inside the span so the turn's first record inherits the
        # trace id the span publishes. ``trace_id_var`` is intentionally not
        # reset on span exit (the request summary is written later).
        turn_token = llm_log.begin_turn(
            run_id=run_id or "",
            model=str(getattr(settings, "openai_model", "") or ""),
            message_chars=len(message),
            history_count=len(history or []),
        )
        try:
            try:
                agent = await assemble_kernel_agent(config, settings)
            except KernelAssemblyError as exc:
                logger.exception("chat kernel assembly failed")
                raise KernelModelFailure(
                    "kernel_unavailable", "Agent 内核暂不可用，请稍后重试。"
                ) from exc

            invoke_config = (
                {"configurable": {"thread_id": f"v2:chat:{run_id}"}}
                if checkpointer is not None
                else None
            )
            # Ticket 04: only tool spans ride on callbacks. Generations are
            # traced by the model decorator (ticket 03), avoiding duplicate
            # generation nodes. LLM_LOG rides the same seam without spans.
            callbacks = _callbacks_for_turn(turn_span, on_activity)
            if callbacks:
                invoke_config = dict(invoke_config or {})
                invoke_config["callbacks"] = callbacks
            # Ticket 05 (T08): ONE input construction for every kernel invocation —
            # bounded history first, the current user message appended exactly once.
            input_messages = build_kernel_messages(message=message, history=history)
            if resume:
                from langgraph.types import Command
                graph_input = Command(resume={resume_interrupt_id: True} if resume_interrupt_id else True)
            else:
                graph_input = {"messages": input_messages}
            try:
                try:
                    if on_text is None:
                        result = await asyncio.wait_for(
                            agent.ainvoke(graph_input, config=invoke_config),
                            timeout=240,
                        )
                        interrupts = result.get("__interrupt__", ())
                    else:
                        result = {}
                        interrupts = ()
                        public_text = _StreamingPublicText()
                        thoughts = _ThoughtBuffer(on_thought)
                        step = None
                        rounds = _ModelRounds(on_activity)
                        try:
                            async with asyncio.timeout(240):
                                async for mode, data in agent.astream(
                                    graph_input, config=invoke_config,
                                    stream_mode=["messages", "values", "updates"],
                                ):
                                    if mode == "values":
                                        result = data
                                        continue
                                    if mode == "updates":
                                        if "__interrupt__" in data:
                                            interrupts = data["__interrupt__"]
                                        continue
                                    chunk, metadata = data
                                    # Compaction, tool models, and subagents are not the reply.
                                    namespace = metadata.get("langgraph_checkpoint_ns", "")
                                    if (metadata.get("langgraph_node") != "model"
                                        or "|" in namespace
                                        or (namespace and not namespace.startswith("model:"))):
                                        continue
                                    # Agent process tracking ticket 03: a main-model
                                    # step boundary opens a new round; the previous
                                    # round is closed by the same edge.
                                    rounds.observe(metadata.get("langgraph_step"))
                                    if getattr(chunk, "tool_call_chunks", None) or getattr(chunk, "tool_calls", None):
                                        # A tool decision ends the round that made it.
                                        rounds.close()
                                        continue
                                    if metadata.get("langgraph_step") != step:
                                        # A new model step opens a new reasoning segment.
                                        await thoughts.flush()
                                        public_text = _StreamingPublicText()
                                        step = metadata.get("langgraph_step")
                                    content = getattr(chunk, "content", "")
                                    if isinstance(content, list):
                                        content = "".join(
                                            block if isinstance(block, str) else block.get("text", "")
                                            for block in content
                                            if isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text")
                                        )
                                    # Some OpenAI-compatible providers return reasoning
                                    # out-of-band instead of inside <think> tags.
                                    extra = getattr(chunk, "additional_kwargs", None)
                                    if isinstance(extra, dict):
                                        for key in ("reasoning_content", "reasoning"):
                                            hidden = extra.get(key)
                                            if isinstance(hidden, str):
                                                await thoughts.add(hidden)
                                    delta = public_text.feed(content if isinstance(content, str) else "")
                                    hidden_delta = public_text.take_reasoning()
                                    if hidden_delta:
                                        await thoughts.add(hidden_delta)
                                    if delta:
                                        await on_text(delta, audit_entries)
                                delta = public_text.feed("", final=True)
                                hidden_delta = public_text.take_reasoning()
                                if hidden_delta:
                                    await thoughts.add(hidden_delta)
                                if delta:
                                    await on_text(delta, audit_entries)
                                await thoughts.flush()
                        finally:
                            # Timeout, model error, interrupt and normal
                            # exhaustion all close the live round exactly once.
                            rounds.close()
                except TimeoutError as exc:
                    raise KernelModelFailure("model_timeout", "模型响应超时，请稍后重试。") from exc
                if interrupts:
                    raise KernelApprovalPending({**interrupts[0].value, "interrupt_id": interrupts[0].id})
                text = extract_final_text(result)
            except KernelApprovalPending:
                raise
            except KernelModelFailure:
                raise
            except Exception as exc:
                mapped = _map_invalid_tool_arguments(exc)
                if mapped is not None:
                    raise mapped from exc
                logger.exception("chat kernel run failed")
                raise KernelModelFailure(
                    "kernel_run_failed",
                    "生成回复失败，请稍后重试。",
                ) from exc
            finally:
                await agent.aclose()

            if not text:
                raise KernelModelFailure("empty_model_output", "未生成有效回复，请重试。")
            tracing.mark_ok(turn_span)
            llm_log.end_turn(turn_token, status="ok", final_text=text)
            return text, audit_entries
        except BaseException as exc:
            # Pause/failure exits still own the audits of completed calls.
            # The orchestrator publishes them before it emits the terminal;
            # without this they die with the local list.
            if isinstance(exc, (KernelApprovalPending, KernelModelFailure)):
                exc.audit_entries = list(audit_entries)
            tracing.record_exception(turn_span, exc)
            llm_log.end_turn(turn_token, status="error", error=exc)
            raise


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
