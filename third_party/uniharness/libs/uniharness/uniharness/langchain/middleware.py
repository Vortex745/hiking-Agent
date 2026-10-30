"""Agent middleware — the actual runtime for UniHarness.

This middleware coordinates compaction, permission gating, skill injection,
image extraction, and system reminder rules within LangChain's agent
infrastructure.

Compaction logic is inlined (no separate controller class). The pre-model
pipeline runs: intercepts -> appenders -> image extraction -> annotators.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware as LangChainAgentMiddleware,
)
from langchain.agents.middleware.types import (
    AgentState,
    ToolCallRequest,
    hook_config,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langgraph.types import Command as LangGraphCommand
from langgraph.types import Overwrite as LangGraphOverwrite

from uniharness.harness import PermissionResult, evaluate_reminders
from uniharness.langchain.adapter import to_langchain_tool
from uniharness.prompts import compose, load, substitute
from uniharness.prompts.tags import SYSTEM_REMINDER_TAG
from uniharness.types import AgentContext, CompactionPhase

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.runtime import Runtime

    from uniharness.harness.environment import EnvironmentResolver
    from uniharness.harness.permission import PermissionGate
    from uniharness.harness.reminders import Reminder
    from uniharness.harness.skills import SkillResolver
    from uniharness.prompts import SectionFn
    from uniharness.types import ApprovalCallback, EnvironmentContext, Skill

logger = logging.getLogger(__name__)


# Minimum messages required before compaction can trigger.
# A single summary message after compaction should not re-trigger.
_MIN_MESSAGES_FOR_COMPACTION = 2

# A run may ask the model to repair malformed tool arguments exactly once.
# A second malformed reply is a stable failure, never another retry.
_MAX_TOOL_ARGUMENT_REPAIRS = 1

# Feedback is deliberately schema-shaped only: it never echoes the malformed
# raw arguments, guessed values, or the model's prompt text.
_TOOL_ARGUMENT_REPAIR_GUIDANCE = (
    "Tool call {name} was rejected before execution because its arguments "
    "are not valid JSON for the tool schema. Re-submit this tool call with "
    "corrected arguments: every string value, including ratios and prompts, "
    "must be quoted with valid JSON double quotes."
)

_SAFE_TOOL_NAME = re.compile(r"[^A-Za-z0-9_.:/-]")


class InvalidToolArgumentsError(RuntimeError):
    """The model's tool-call arguments cannot be executed as written.

    Carries only the internal failure code, the number of invalid calls, and
    their call ids — never raw arguments, prompts, or exception text. The
    kernel maps this to the public ``invalid_tool_arguments`` failure.
    """

    def __init__(
        self,
        code: str,
        *,
        call_count: int,
        call_ids: Sequence[str] = (),
    ) -> None:
        super().__init__(code)
        self.code = code
        self.call_count = call_count
        self.call_ids = tuple(call_ids)


def _safe_tool_name(value: Any) -> str:
    """Redact a tool name for logs (length and charset bounded)."""
    text = str(value) if value else "unknown"
    return _SAFE_TOOL_NAME.sub("_", text)[:80]


def _extract_text_content(content: str | list[Any]) -> str:
    """Extract text from LangChain message content (str or list[block])."""
    if isinstance(content, str):
        return content
    return "".join(block if isinstance(block, str) else block.get("text", "") for block in content if isinstance(block, (str, dict)))


def _rebuild_message(msg: BaseMessage, new_content: str) -> BaseMessage:
    """Reconstruct a message with modified content, preserving metadata."""
    kwargs: dict[str, Any] = {"content": new_content}
    if msg.id is not None:
        kwargs["id"] = msg.id
    if isinstance(msg, ToolMessage):
        kwargs["tool_call_id"] = msg.tool_call_id
    return msg.__class__(**kwargs)


def _create_denied_response(
    request: ToolCallRequest,
    reason: str | None,
) -> ToolMessage:
    """Create a tool response for a denied action."""
    error_message = f"Permission denied: {reason}" if reason else "Permission denied"
    return ToolMessage(
        content=error_message,
        tool_call_id=request.tool_call.get("id", ""),
    )


def _supports_tool_images(model: BaseChatModel) -> bool:
    """Check if the model's provider supports images in tool result messages.

    ``ChatAnthropic`` (and Anthropic-compatible providers using the same
    class) natively supports image content blocks inside ``tool_result``
    messages.  Other providers (OpenAI, DeepSeek, etc.) reject them.
    """
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return False
    return isinstance(model, ChatAnthropic)


# Marker key in ``additional_kwargs`` to prevent double extraction.
_IMAGE_EXTRACTED = "uniharness:images_extracted"


def _extract_tool_images(messages: Sequence[BaseMessage]) -> list[BaseMessage] | None:
    """Move images from ToolMessages into follow-up HumanMessages.

    For providers that reject images in tool-role messages (OpenAI),
    this extracts image content blocks and injects them as user messages
    immediately after the originating ToolMessage.

    Already-processed messages (marked via ``additional_kwargs``) are
    skipped so the function is safe to call repeatedly.

    Returns a new list if any extraction happened, ``None`` otherwise.
    """
    result: list[BaseMessage] = []
    changed = False

    for msg in messages:
        result.append(msg)

        if not isinstance(msg, ToolMessage):
            continue
        if not isinstance(msg.content, list):
            continue
        if msg.additional_kwargs.get(_IMAGE_EXTRACTED):
            continue

        text_blocks: list[str | dict[str, Any]] = []
        image_blocks: list[dict[str, Any]] = []
        for block in msg.content:
            if isinstance(block, dict) and block.get("type") in ("image", "image_url"):
                image_blocks.append(block)
            else:
                text_blocks.append(block)

        if not image_blocks:
            continue

        changed = True

        # Replace ToolMessage content: text blocks only, or a placeholder string.
        tool_content: str | list[str | dict[str, Any]] = text_blocks if text_blocks else "[see image below]"
        result[-1] = ToolMessage(
            content=tool_content,
            tool_call_id=msg.tool_call_id,
            id=msg.id,
            additional_kwargs={**msg.additional_kwargs, _IMAGE_EXTRACTED: True},
        )

        # Inject HumanMessage with extracted images
        result.append(
            HumanMessage(
                content=[
                    {"type": "text", "text": "[Image from tool result]"},
                    *image_blocks,
                ]
            )
        )

    return result if changed else None


def _detect_skill_call(openai_msgs: Sequence[dict[str, Any]]) -> str | None:
    """Detect if the most recent tool call was a Skill invocation.

    Operates on OpenAI-format messages (dicts with ``role``, ``tool_calls``,
    ``tool_call_id``, etc.).

    Returns the skill name, or ``None`` if no recent skill tool call found
    or if a user message already follows (already injected).
    """
    from uniharness.tools.skill import SkillTool

    skill_tool_name = SkillTool.name

    if not openai_msgs:
        return None

    # Walk backwards: find the last tool-role message.
    # If we hit a user message first, the skill was already injected.
    last_tool_idx: int | None = None
    for i in range(len(openai_msgs) - 1, -1, -1):
        role = openai_msgs[i].get("role")
        if role == "user":
            return None
        if role == "tool":
            last_tool_idx = i
            break

    if last_tool_idx is None:
        return None

    tool_call_id = openai_msgs[last_tool_idx].get("tool_call_id")

    # Find the corresponding assistant message with tool_calls.
    for i in range(last_tool_idx - 1, -1, -1):
        msg = openai_msgs[i]
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            if tc.get("id") == tool_call_id and fn.get("name") == skill_tool_name:
                import json

                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}
                result: str | None = args.get("skill")
                return result
        break  # Only check the nearest assistant message

    return None


class UniHarnessState(AgentState):
    """Extended agent state with compaction and tool-argument repair tracking.

    Both repair fields are optional so checkpoints written before this
    contract remain readable: a missing count means "no repair used yet",
    a missing flag means "not waiting for a repair".
    """

    compaction_phase: NotRequired[CompactionPhase]
    invalid_tool_call_repair_count: NotRequired[int]
    awaiting_tool_call_repair: NotRequired[bool]


class AgentMiddleware(LangChainAgentMiddleware):
    """The UniHarness runtime, implemented as LangChain middleware.

    Coordinates:
    - System prompt injection (before-agent, once per invocation)
    - Compaction (3-phase inlined state machine)
    - Skill injection (appender)
    - Reminder rules (annotators)
    - Permission gating (tool call wrapper)

    Pre-model pipeline runs ordered groups:
    1. Intercepts: compaction phases (may abort normal processing)
    2. Appenders: skill injection (adds messages)
    3. Annotators: reminder rules (injects into last message)
    """

    state_schema = UniHarnessState

    def __init__(
        self,
        *,
        context: AgentContext,
        system_prompt: str,
        permission_gate: PermissionGate,
        approval_callback: ApprovalCallback | None = None,
        environment_resolver: EnvironmentResolver | None = None,
        skill_resolver: SkillResolver | None = None,
        reminders: Sequence[Reminder] = (),
        prompt_profile: Sequence[SectionFn] | None = None,
        custom_prompt: str = "",
    ) -> None:
        """Initialize the middleware.

        Args:
            context: Frozen agent context snapshot.
            system_prompt: Assembled system prompt for first-turn injection.
            permission_gate: Permission checking for tool calls.
            approval_callback: Human-in-the-loop approval callback.
            environment_resolver: Resolves runtime environment on compaction.
            skill_resolver: Resolves skills for injection and compaction.
            reminders: Reminder rules for dynamic system-reminder injection.
            prompt_profile: Which section profile to recompose on compaction
                rebuild. ``None`` means fully custom prompt (no recomposition).
            custom_prompt: Developer's custom prompt prefix (e.g.
                ``definition.system_prompt`` for subagents). Prepended to
                framework-composed sections on compaction rebuild.
        """
        self._context = context
        self._system_prompt = system_prompt
        self._permission_gate = permission_gate
        self._approval_callback = approval_callback
        self._environment_resolver = environment_resolver
        self._skill_resolver = skill_resolver
        self._reminders = list(reminders)
        self._prompt_profile = prompt_profile
        self._custom_prompt = custom_prompt
        self._compaction_prompt = load("user_prompt_compaction_request")
        self._summary_template = load("user_prompt_compaction_summary_rebuild")
        self._tools_cache: list[BaseTool] | None = None

    @property
    def tools(self) -> Sequence[BaseTool]:  # type: ignore[override]
        """Get tools as LangChain tools (cached)."""
        if self._tools_cache is None:
            fmt: Literal["anthropic", "openai"] = "anthropic" if _supports_tool_images(self._context.model.model) else "openai"
            self._tools_cache = [to_langchain_tool(tool, content_format=fmt) for tool in self._context.tools]
        return self._tools_cache

    @staticmethod
    def _get_total_tokens(messages: Sequence[BaseMessage]) -> int | None:
        """Extract total_tokens from the last AIMessage's usage_metadata.

        Returns None if unavailable (e.g. fake models in tests).
        """
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                metadata = msg.usage_metadata
                if metadata is not None:
                    return metadata["total_tokens"]
                return None
        return None

    async def _rebuild_after_compaction(self, summary: str) -> list[BaseMessage]:
        """Rebuild messages after compaction.

        Three cases based on ``prompt_profile``:
        - ``None`` → fully custom prompt, no recomposition
        - ``custom_prompt`` non-empty → prepend to framework sections (subagent)
        - Both empty/default → pure framework recomposition (root)
        """
        summary_content = substitute(self._summary_template, SUMMARY_CONTENT=summary)

        # Case: fully custom prompt — no recomposition
        if self._prompt_profile is None:
            return [
                SystemMessage(content=self._system_prompt),
                HumanMessage(content=summary_content),
            ]

        # Cases: root (no custom) or subagent — recompose framework sections
        env: EnvironmentContext | None
        if self._environment_resolver is not None:
            env = await self._environment_resolver.resolve()
        else:
            env = self._context.environment

        if self._skill_resolver is not None:
            skills: list[Skill] = await self._skill_resolver.discover()
        else:
            skills = list(self._context.skills)

        self._context = replace(self._context, environment=env, skills=skills)
        framework_prompt = compose(self._prompt_profile, self._context)

        if self._custom_prompt:
            self._system_prompt = f"{self._custom_prompt}\n\n{framework_prompt}"
        else:
            self._system_prompt = framework_prompt

        return [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=summary_content),
        ]

    # --- Async hooks (primary implementations) ---

    async def abefore_agent(
        self,
        state: AgentState,
        _runtime: Runtime[Any] | None = None,
    ) -> dict[str, Any] | None:
        """One-time setup before agent execution starts.

        Injects the system prompt if not already present.
        """
        messages = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=self._system_prompt), *messages]
            return {"messages": LangGraphOverwrite(messages)}
        return None

    async def abefore_model(
        self,
        state: AgentState,
        _runtime: Runtime[Any] | None = None,
    ) -> dict[str, Any] | None:
        """Pre-model pipeline.

        Group 1: Intercepts (compaction phases, early exit).
        Group 2: Appenders (skill injection).
        Group 2.5: Image extraction (non-Anthropic providers).
        Group 3: Annotators (system reminders).
        """
        messages: list[BaseMessage] = list(state["messages"])

        # --- GROUP 1: Intercepts (compaction phases) ---
        phase = CompactionPhase(state.get("compaction_phase", CompactionPhase.NONE))

        if phase == CompactionPhase.REQUESTING:
            appended = [HumanMessage(content=self._compaction_prompt)]
            return {
                "messages": appended,
                "compaction_phase": phase,
            }

        if phase == CompactionPhase.APPLYING:
            last = messages[-1]
            if not isinstance(last, AIMessage):
                msg = (
                    f"Compaction state machine bug: APPLYING phase requires "
                    f"the model's summary as the last message (AIMessage), "
                    f"but got {type(last).__name__}. "
                )
                raise TypeError(msg)
            summary = _extract_text_content(last.content)
            rebuilt = await self._rebuild_after_compaction(summary)
            return {
                "messages": LangGraphOverwrite(rebuilt),
                "compaction_phase": CompactionPhase.NONE,
            }

        # --- GROUP 2: Appenders (skill injection) ---
        if self._skill_resolver is not None:
            openai_msgs = convert_to_openai_messages(messages)
            skill_name = _detect_skill_call(openai_msgs)

            if skill_name is not None:
                try:
                    content = await self._skill_resolver.load_content(skill_name)
                except (KeyError, RuntimeError) as exc:
                    template = load("system_reminder_skill_launch_failure")
                    content = SYSTEM_REMINDER_TAG(
                        substitute(template, SKILL_NAME=skill_name, FAILURE_MSG=repr(exc)),
                    )
                appended = [HumanMessage(content=content)]
                return {
                    "messages": appended,
                }

        # --- GROUP 2.5: Image extraction (non-Anthropic only) ---
        images_extracted = False
        if not _supports_tool_images(self._context.model.model):
            extracted = _extract_tool_images(messages)
            if extracted is not None:
                logger.info("Extracted images from %d ToolMessage(s) into HumanMessage(s)", len(extracted) - len(messages))
                messages = extracted
                images_extracted = True

        # --- GROUP 3: Annotators (system reminders) ---
        if self._reminders:
            openai_msgs = convert_to_openai_messages(messages)
            prepends, appends = evaluate_reminders(
                self._reminders,
                openai_msgs,
                self._context,
            )

            if prepends or appends:
                last_msg = messages[-1]
                content_str = _extract_text_content(last_msg.content)

                reminder_parts = [*prepends, content_str, *appends]
                new_content = "\n\n".join(part for part in reminder_parts if part)

                patched = [*messages[:-1], _rebuild_message(last_msg, new_content)]
                return {
                    "messages": LangGraphOverwrite(patched),
                }

        if images_extracted:
            return {"messages": LangGraphOverwrite(messages)}

        return None

    def _check_invalid_tool_calls(self, state: AgentState) -> dict[str, Any] | None:
        """Repair malformed tool arguments once, or fail before any execution.

        LangChain's model-to-tools edge only counts valid ``tool_calls``; a
        message whose arguments failed JSON parsing therefore ends the loop
        unless middleware answers the invalid calls. This check runs before
        compaction and before tools, so a mixed or unrepairable message can
        never execute any call from that message.
        """
        messages = state.get("messages") or []
        if not messages:
            return None
        last = messages[-1]
        if not isinstance(last, AIMessage):
            return None

        invalid = list(last.invalid_tool_calls or ())
        valid = list(last.tool_calls or ())
        awaiting = bool(state.get("awaiting_tool_call_repair", False))
        repaired = int(state.get("invalid_tool_call_repair_count", 0) or 0)

        if not invalid:
            if not awaiting:
                return None
            if valid:
                # The repair produced executable calls: resume the normal
                # schema / approval / effect path with the budget consumed.
                return {"awaiting_tool_call_repair": False}
            # The repair reply contains neither a tool call nor a replacement
            # call: it is preamble-only, not a successful completion.
            logger.warning("tool-argument repair reply had no tool call")
            raise InvalidToolArgumentsError(
                "tool_call_arguments_repair_failed",
                call_count=0,
            )

        call_ids = tuple(str(call["id"]) for call in invalid if call.get("id"))
        tool_names = ",".join(_safe_tool_name(call.get("name")) for call in invalid)
        logger.warning(
            "invalid tool arguments: calls=%d tools=%s ids=%s repaired=%d awaiting=%s",
            len(invalid),
            tool_names,
            ",".join(call_ids) or "-",
            repaired,
            awaiting,
        )

        if valid:
            raise InvalidToolArgumentsError(
                "mixed_tool_call_arguments",
                call_count=len(invalid),
                call_ids=call_ids,
            )
        if awaiting or repaired >= _MAX_TOOL_ARGUMENT_REPAIRS:
            raise InvalidToolArgumentsError(
                "tool_call_arguments_repair_exhausted",
                call_count=len(invalid),
                call_ids=call_ids,
            )
        if len(call_ids) != len(invalid):
            # Without an id there is no way to answer the call; the model
            # cannot be asked to repair it safely.
            raise InvalidToolArgumentsError(
                "tool_call_arguments_missing_id",
                call_count=len(invalid),
                call_ids=call_ids,
            )

        feedback = [
            ToolMessage(
                content=_TOOL_ARGUMENT_REPAIR_GUIDANCE.format(
                    name=_safe_tool_name(call.get("name")),
                ),
                tool_call_id=call_id,
                status="error",
            )
            for call, call_id in zip(invalid, call_ids, strict=True)
        ]
        return {
            "messages": feedback,
            "invalid_tool_call_repair_count": repaired + 1,
            "awaiting_tool_call_repair": True,
            "jump_to": "model",
        }

    @hook_config(can_jump_to=["model"])
    async def aafter_model(
        self,
        state: AgentState,
        _runtime: Runtime[Any] | None = None,
    ) -> dict[str, Any] | None:
        """Repair invalid tool arguments, then check the compaction trigger."""
        messages = state["messages"]
        phase = CompactionPhase(state.get("compaction_phase", CompactionPhase.NONE))

        # Compaction turns are summaries, not tool decisions; their repair
        # state is left untouched and the state machine below owns them.
        if phase == CompactionPhase.NONE:
            repair = self._check_invalid_tool_calls(state)
            if repair is not None:
                return repair

        # Trigger compaction if threshold exceeded
        if phase == CompactionPhase.NONE and len(messages) >= _MIN_MESSAGES_FOR_COMPACTION:
            token_count = self._get_total_tokens(messages)
            threshold = self._context.model.compaction_threshold
            assert threshold is not None  # noqa: S101  # guaranteed by _resolve_to_profile
            if token_count is not None and token_count >= threshold:
                return {"compaction_phase": CompactionPhase.REQUESTING, "jump_to": "model"}

        # Advance state machine: LLM just generated the summary
        if phase == CompactionPhase.REQUESTING:
            return {"compaction_phase": CompactionPhase.APPLYING, "jump_to": "model"}

        return None

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | LangGraphCommand[Any]]],
    ) -> ToolMessage | LangGraphCommand[Any]:
        """Check permission before tool execution."""
        tool_name = request.tool_call["name"]
        tool_args = request.tool_call.get("args", {})

        decision = await self._permission_gate.check(tool_name, tool_args)

        if decision.result == PermissionResult.DENIED:
            return _create_denied_response(request, decision.reason)

        if decision.result == PermissionResult.NEEDS_APPROVAL:
            if self._approval_callback is None:
                return _create_denied_response(
                    request,
                    f"Action requires approval: {decision.approval_prompt or 'No details provided'}",
                )

            approved = await self._approval_callback(
                tool_name,
                tool_args,
                decision.approval_prompt,
            )

            if not approved:
                return _create_denied_response(request, "Action denied by user")

        return await handler(request)
