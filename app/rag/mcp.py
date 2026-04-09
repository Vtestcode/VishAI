"""
Helpers for optional remote MCP integration through the OpenAI Responses API.
"""

from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from app.core.cache import build_cache_key, cache_get_json, cache_set_json
from app.core.config import Settings
from app.models.schemas import ToolCall, ToolDefinition

DEFAULT_ROUTABLE_TOOLS = {
    "search_knowledge_base",
    "web_search",
    "get_current_datetime",
    "explore_public_repo_readmes",
    "search_github_code",
}

STRICT_ROUTE_PATTERNS: dict[str, tuple[str, ...]] = {
    "get_current_datetime": (
        "what time is it",
        "current time",
        "what is the date",
        "current date",
        "date today",
        "right now",
        "current datetime",
    ),
    "web_search": (
        "search the web",
        "web search",
        "look up online",
        "find online",
        "latest news",
        "latest update",
        "latest release",
        "recent news",
    ),
    "explore_public_repo_readmes": (
        "readme",
        "repo readme",
        "repo readmes",
        "repository readme",
        "repository readmes",
        "public repo readmes",
        "public repositories",
    ),
    "search_github_code": (
        "github code",
        "search github code",
        "search github for code",
        "code in github",
        "github implementation",
        "github example",
    ),
    "search_knowledge_base": (
        "search the knowledge base",
        "search your knowledge base",
        "what does the knowledge base say",
        "search the docs",
        "based on the docs",
        "portfolio documents",
    ),
}


def mcp_enabled(settings: Settings) -> bool:
    """Return True when the app should expose a remote MCP server."""
    return bool(settings.enable_mcp and settings.mcp_server_url and settings.openai_api_key)


def build_mcp_tool_config(
    settings: Settings,
    allowed_tools_override: list[str] | None = None,
) -> dict[str, Any] | None:
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

    allowed_tools = allowed_tools_override or [
        item.strip()
        for item in settings.mcp_allowed_tools.split(",")
        if item.strip()
    ]
    if allowed_tools:
        tool["allowed_tools"] = allowed_tools

    return tool


def route_query_to_tool(question: str, settings: Settings) -> str | None:
    """Choose the most likely MCP tool for a user question."""
    allowed_tools = {
        item.strip()
        for item in settings.mcp_allowed_tools.split(",")
        if item.strip()
    } or DEFAULT_ROUTABLE_TOOLS

    text = question.lower().strip()

    def available(name: str) -> bool:
        return name in allowed_tools

    if available("get_current_datetime") and (
        any(phrase in text for phrase in STRICT_ROUTE_PATTERNS["get_current_datetime"])
        or re.search(r"\b(what time|current time|today's date|date today|right now)\b", text)
    ):
        return "get_current_datetime"

    if available("explore_public_repo_readmes") and (
        any(phrase in text for phrase in STRICT_ROUTE_PATTERNS["explore_public_repo_readmes"])
        or (
            "github" in text and any(term in text for term in ("repos", "repositories", "repo summaries", "repo readmes"))
        )
    ):
        return "explore_public_repo_readmes"

    if available("search_github_code") and (
        any(phrase in text for phrase in STRICT_ROUTE_PATTERNS["search_github_code"])
        or "search github" in text
        or ("github" in text and "code" in text)
        or ("repo" in text and "code" in text)
        or "example implementation" in text
    ):
        return "search_github_code"

    if available("web_search") and (
        any(phrase in text for phrase in STRICT_ROUTE_PATTERNS["web_search"])
        or (
            any(term in text for term in ("latest", "news", "recent", "online"))
            and not any(term in text for term in ("vishal", "portfolio", "resume", "experience", "project"))
        )
    ):
        return "web_search"

    if available("search_knowledge_base") and (
        any(phrase in text for phrase in STRICT_ROUTE_PATTERNS["search_knowledge_base"])
        or (
            any(term in text for term in ("vishal", "portfolio", "resume"))
            and any(term in text for term in ("docs", "documents", "knowledge base"))
        )
    ):
        return "search_knowledge_base"

    return None


def should_skip_retrieval(question: str, settings: Settings) -> tuple[bool, str | None]:
    """Return True when a question should bypass RAG retrieval and use a tool directly."""
    routed_tool = route_query_to_tool(question, settings)
    if routed_tool in {
        "get_current_datetime",
        "web_search",
        "explore_public_repo_readmes",
        "search_github_code",
    }:
        return True, routed_tool
    return False, routed_tool


def fetch_available_tools(settings: Settings) -> list[ToolDefinition]:
    """Ask OpenAI to discover the remote MCP server tools and return them."""
    cache_key = build_cache_key(
        "mcp-tools",
        f"{settings.mcp_server_label}|{settings.mcp_server_url}|{settings.mcp_allowed_tools}",
    )
    cached = cache_get_json(cache_key)
    if isinstance(cached, list):
        return [ToolDefinition(**item) for item in cached]

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
    tools = _extract_available_tools(payload)
    cache_set_json(
        cache_key,
        [tool_item.model_dump() for tool_item in tools],
        settings.redis_cache_ttl_seconds,
    )
    return tools


def answer_with_mcp(
    *,
    system_prompt: str,
    user_prompt: str,
    settings: Settings,
    preferred_tool: str | None = None,
) -> tuple[str, list[ToolDefinition], list[ToolCall]]:
    """Generate an answer with optional remote MCP access."""
    tool = build_mcp_tool_config(
        settings,
        allowed_tools_override=[preferred_tool] if preferred_tool else None,
    )
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
