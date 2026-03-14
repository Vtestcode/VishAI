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


class ChatResponse(BaseModel):
    """Response body for POST /chat."""
    answer: str
    sources: list[SourceSnippet] = Field(default_factory=list)


# ── Ingest ──────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    """Response body for POST /ingest."""
    status: str = "ok"
    documents_ingested: int = 0
    message: str = ""


# ── Health ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: str = "ok"
