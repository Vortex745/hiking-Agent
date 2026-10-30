"""Product adapters for tested MCP tools that execute directly."""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
import uuid
from typing import Any, Callable

from mcp.client.auth import OAuthFlowError, OAuthRegistrationError, OAuthTokenError
from pydantic import BaseModel
from uniharness.mcp._schema import json_schema_to_model
from uniharness.tools.base import BaseAgentTool
from uniharness.types import ToolResult, UrlSource

from agent_v2.approvals import canonical_args
from agent_v2.effects import (
    DEFAULT_EFFECT_CLAIM_TTL_SECONDS,
    canonical_effect_status,
    compute_effect_identity,
)
from agent_v2.mcp_config_store import LocalMcpClient, client_config, load_servers
from agent_v2.mcp_oauth_store import oauth_auth

logger = logging.getLogger(__name__)


def _exception_chain(exc: BaseException) -> str:
    names: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        names.append(type(current).__name__)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return "->".join(names)


def _redacted_traceback(exc: BaseException, server: dict) -> str:
    """Stack frames only (no exception message) with secrets replaced."""
    text = "".join(traceback.format_tb(exc.__traceback__))
    for secret in (*server.get("headers", {}).values(), *server.get("env", {}).values()):
        if secret:
            text = text.replace(str(secret), "[已隐藏]")
    return text


def current_mcp_call_id() -> str:
    from langgraph.config import get_config

    return get_config()["metadata"]["langgraph_checkpoint_ns"]


def _is_oauth_error(exc: BaseException) -> bool:
    if isinstance(exc, (OAuthFlowError, OAuthRegistrationError, OAuthTokenError)):
        return True
    return any(_is_oauth_error(inner) for inner in getattr(exc, "exceptions", ()))


class CustomMcpTool(BaseAgentTool[Any]):
    def __init__(self, *, server: dict, descriptor: dict, revision: str,
                 run_id: str, runs_repo: Any, audit_sink: Callable,
                 owner_id: str = "") -> None:
        self.name = f"mcp__{server['id']}__{descriptor['name']}"
        self.description = descriptor["description"]
        self.args_schema = json_schema_to_model(self.name, descriptor["schema"])
        self._server_id = server["id"]
        self._revision = revision
        self._tool_name = descriptor["name"]
        self._run_id = run_id
        self._runs_repo = runs_repo
        self._audit_sink = audit_sink
        self._owner_id = owner_id

    async def execute(self, params: BaseModel) -> ToolResult:
        args = params.model_dump(exclude_unset=True)
        call_args = {"call_id": current_mcp_call_id(), "arguments": args}
        run_record = self._runs_repo.get_run(self._run_id)
        server = next(
            (item for item in load_servers(self._owner_id or None) if item["id"] == self._server_id),
            None,
        )
        if server is None or not server["enabled"] or server["revision"] != self._revision:
            return ToolResult(error="MCP 配置已改变，请重新发起请求。")

        identity = compute_effect_identity(run_id=self._run_id, tool_name=self.name, args=call_args)
        effect = self._runs_repo.get_effect(identity.effect_id)
        if effect is None:
            effect = self._runs_repo.insert_effect(
                effect_key=identity.effect_id, run_id=self._run_id,
                conversation_id=run_record.conversation_id if run_record else "",
                tool_name=self.name,
                args_canonical=canonical_args(call_args),
                external_idempotency_key=identity.idempotency_key,
            )
        if canonical_effect_status(effect.status) == "committed":
            stored = json.loads(effect.result_json)
            return ToolResult(output=stored.get("output"), error=stored.get("error"))
        if canonical_effect_status(effect.status) == "claimed":
            self._runs_repo.mark_effect_outcome_unknown(identity.effect_id, reason="mcp_interrupted")
            return ToolResult(error="上次 MCP 调用结果未知，未自动重试。")
        claim_token = uuid.uuid4().hex
        _, won = self._runs_repo.claim_effect(
            identity.effect_id, owner=f"mcp:{self._run_id}", claim_token=claim_token,
            ttl_seconds=DEFAULT_EFFECT_CLAIM_TTL_SECONDS,
        )
        if not won:
            return ToolResult(error="MCP 调用无法取得执行权，未执行。")
        try:
            auth = await oauth_auth(server, self._owner_id or None)
            async with asyncio.timeout(15):
                async with LocalMcpClient(self._server_id, client_config(server), auth=auth) as client:
                    tool = next(item for item in client.tools if item.name == self.name)
                    result = await tool.execute(params)
        except asyncio.CancelledError as exc:
            self._runs_repo.mark_effect_outcome_unknown(
                identity.effect_id, reason="mcp_call_interrupted", claim_token=claim_token
            )
            logger.warning(
                "custom_mcp: call cancelled run_id=%s tool=%s effect_id=%s "
                "error_type=%s chain=%s",
                self._run_id,
                self.name,
                identity.effect_id,
                type(exc).__name__,
                _exception_chain(exc),
            )
            raise
        except Exception as exc:
            if _is_oauth_error(exc):
                # Authorization failed before the tool executed: this is a
                # known failure, not an indeterminate side effect. Surface
                # the actionable re-authorization path instead of "unknown".
                message = (
                    "MCP 授权已过期或失效，请在 设置 → MCP 接入 中"
                    "重新测试并完成授权后重试。"
                )
                logger.warning(
                    "custom_mcp: authorization failed run_id=%s tool=%s "
                    "effect_id=%s error_type=%s chain=%s\n%s",
                    self._run_id,
                    self.name,
                    identity.effect_id,
                    type(exc).__name__,
                    _exception_chain(exc),
                    _redacted_traceback(exc, server),
                )
                self._runs_repo.mark_effect_failed(
                    identity.effect_id,
                    reason="mcp_oauth_reauth_required",
                    claim_token=claim_token,
                )
                self._audit_sink({"tool": self.name, "run_id": self._run_id, "source": "mcp",
                                  "server_id": self._server_id, "phase": "result", "ok": False,
                                  "code": "mcp_oauth_reauth_required",
                                  "message": message})
                return ToolResult(error=message)
            logger.error(
                "custom_mcp: call failed run_id=%s tool=%s effect_id=%s "
                "error_type=%s chain=%s\n%s",
                self._run_id,
                self.name,
                identity.effect_id,
                type(exc).__name__,
                _exception_chain(exc),
                _redacted_traceback(exc, server),
            )
            self._runs_repo.mark_effect_outcome_unknown(
                identity.effect_id, reason="mcp_call_interrupted", claim_token=claim_token
            )
            self._audit_sink({"tool": self.name, "run_id": self._run_id, "source": "mcp",
                              "server_id": self._server_id, "phase": "result", "ok": False,
                              "code": "outcome_unknown", "status": "outcome_unknown",
                              "message": "本次调用结果未确认"})
            return ToolResult(error="本次调用结果未确认，未自动重试。")

        output, error = result.output or "", result.error or ""
        image_lines = [
            f"![MCP 图片]({image.url})"
            for image in result.images
            if isinstance(image, UrlSource) and image.url.startswith(("https://", "http://"))
        ]
        if image_lines:
            output = "\n".join(part for part in [output, *image_lines] if part)
        for secret in (*server["headers"].values(), *server["env"].values()):
            if secret:
                output, error = output.replace(secret, "[已隐藏]"), error.replace(secret, "[已隐藏]")
        output, error = output[:65536], error[:65536]
        committed = self._runs_repo.commit_effect(
            identity.effect_id, json.dumps({"output": output, "error": error}, ensure_ascii=False),
            claim_token=claim_token, fencing_token=0,
        )
        if committed is None or canonical_effect_status(committed.status) != "committed":
            logger.error(
                "custom_mcp: commit failed run_id=%s tool=%s effect_id=%s",
                self._run_id,
                self.name,
                identity.effect_id,
            )
            self._runs_repo.mark_effect_outcome_unknown(identity.effect_id, reason="mcp_commit_failed", claim_token=claim_token)
            self._audit_sink({"tool": self.name, "run_id": self._run_id, "source": "mcp",
                              "server_id": self._server_id, "phase": "result", "ok": False,
                              "code": "outcome_unknown", "status": "outcome_unknown",
                              "message": "本次调用结果未确认"})
            return ToolResult(error="MCP 调用结果无法确认，未自动重试。")
        self._audit_sink({"tool": self.name, "run_id": self._run_id, "source": "mcp",
                          "server_id": self._server_id, "phase": "result", "ok": not error,
                          "message": output if not error else error})
        return ToolResult(output=output or None, error=error or None)


def build_custom_mcp_tools(
    projection: Any, runs_repo: Any, audit_sink: Callable, owner_id: str = ""
) -> list[CustomMcpTool]:
    servers = {
        server["id"]: server for server in load_servers(owner_id or None)
    }
    tools: list[CustomMcpTool] = []
    for service in projection.mcp_services:
        if not service.revision:
            continue
        server = servers.get(service.server_id)
        if server is None or not server["enabled"] or server["revision"] != service.revision:
            raise ValueError("MCP 配置已改变，请重新发起请求。")
        for descriptor in server["tool_descriptors"]:
            if descriptor["name"] in service.tools:
                tools.append(CustomMcpTool(
                    server=server, descriptor=descriptor, revision=service.revision,
                    run_id=projection.run_id, runs_repo=runs_repo, audit_sink=audit_sink,
                    owner_id=owner_id,
                ))
    return tools
