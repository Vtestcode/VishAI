"""
Helpers for optional remote MCP integration through the OpenAI Responses API.
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from app.core.config import Settings
from app.models.schemas import ToolCall, ToolDefinition


def mcp_enabled(settings: Settings) -> bool:
    """Return True when the app should expose a remote MCP server."""
    return bool(settings.enable_mcp and settings.mcp_server_url and settings.openai_api_key)


def build_mcp_tool_config(settings: Settings) -> dict[str, Any] | None:
    """Build the OpenAI Responses API tool definition for a remote MCP server."""
    if not mcp_enabled(settings):
        return None

    tool: dict[str, Any] = {
        "type": "mcp",
        "server_label": settings.mcp_server_label,
        "server_url": settings.mcp_server_url,
        "require_approval": settings.mcp_require_approval,
    }

    if settings.mcp_server_description:
        tool["server_description"] = settings.mcp_server_description

    allowed_tools = [
        item.strip()
        for item in settings.mcp_allowed_tools.split(",")
        if item.strip()
    ]
    if allowed_tools:
        tool["allowed_tools"] = allowed_tools

    return tool


def fetch_available_tools(settings: Settings) -> list[ToolDefinition]:
    """Ask OpenAI to discover the remote MCP server tools and return them."""
    tool = build_mcp_tool_config(settings)
    if tool is None:
        return []

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.responses.create(
        model=settings.reasoning_model_name,
        tools=[tool],
        input="List the tools that are available from the connected MCP server.",
        max_output_tokens=120,
    )
    payload = _response_to_dict(response)
    return _extract_available_tools(payload)


def answer_with_mcp(
    *,
    system_prompt: str,
    user_prompt: str,
    settings: Settings,
) -> tuple[str, list[ToolDefinition], list[ToolCall]]:
    """Generate an answer with optional remote MCP access."""
    tool = build_mcp_tool_config(settings)
    if tool is None:
        return "", [], []

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.responses.create(
        model=settings.reasoning_model_name,
        tools=[tool],
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    payload = _response_to_dict(response)
    answer = (payload.get("output_text") or getattr(response, "output_text", "") or "").strip()
    return answer, _extract_available_tools(payload), _extract_tool_calls(payload)


def _response_to_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if isinstance(response, dict):
        return response
    return json.loads(json.dumps(response, default=str))


def _extract_available_tools(payload: dict[str, Any]) -> list[ToolDefinition]:
    tools: list[ToolDefinition] = []

    for item in payload.get("output", []):
        if item.get("type") != "mcp_list_tools":
            continue

        for tool in item.get("tools", []):
            tools.append(
                ToolDefinition(
                    name=str(tool.get("name", "")),
                    description=str(tool.get("description", "") or ""),
                    input_schema=tool.get("input_schema", tool.get("inputSchema", {})) or {},
                )
            )

    return tools


def _extract_tool_calls(payload: dict[str, Any]) -> list[ToolCall]:
    calls: list[ToolCall] = []

    for item in payload.get("output", []):
        if item.get("type") != "mcp_call":
            continue

        arguments = item.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}

        if not isinstance(arguments, dict):
            arguments = {}

        calls.append(
            ToolCall(
                name=str(item.get("name", item.get("tool_name", ""))),
                arguments=arguments,
            )
        )

    return calls
