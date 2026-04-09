"""
Pydantic request / response models for the API.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Chat ────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Body of POST /chat."""
    message: str = Field(..., min_length=1, description="User's question")
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session identifier for conversation continuity",
    )


class SourceSnippet(BaseModel):
    """A single retrieved source chunk returned alongside an answer."""
    text: str = Field(..., description="Relevant text excerpt")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk metadata (e.g. file name, page number)",
    )


class ToolDefinition(BaseModel):
    """A user-visible tool exposed by a connected MCP server."""
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """A tool call made during a chat response."""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """Response body for POST /chat."""
    answer: str
    sources: list[SourceSnippet] = Field(default_factory=list)
    session_id: Optional[str] = None
    available_tools: list[ToolDefinition] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    routed_tool: Optional[str] = None


class ToolsResponse(BaseModel):
    """Response body for GET /tools."""
    enabled: bool = False
    server_label: str = ""
    tools: list[ToolDefinition] = Field(default_factory=list)


# ── Ingest ──────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    """Response body for POST /ingest."""
    status: str = "ok"
    documents_ingested: int = 0
    mode: str = "incremental"
    message: str = ""


# ── Health ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: str = "ok"
