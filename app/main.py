"""
FastAPI application entry point.

Registers all API routers, serves static files, and renders
the chat UI templates.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes_chat import router as chat_router
from app.api.routes_ingest import router as ingest_router
from app.api.routes_health import router as health_router

# ── Logging ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── App creation ────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Chatbot",
    description="Retrieval-Augmented Generation chatbot powered by FastAPI, LangChain, Chroma, and OpenAI.",
    version="1.0.0",
)

# ── CORS (allow everything for development / iframe embedding) ──────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files & templates ────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent  # project root

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ── API routers ─────────────────────────────────────────────────────────

app.include_router(chat_router)
app.include_router(ingest_router)
app.include_router(health_router)

# ── HTML pages ──────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Full-page chat UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/widget", response_class=HTMLResponse)
async def widget(request: Request) -> HTMLResponse:
    """Compact chat widget for iframe embedding in Google Sites."""
    return templates.TemplateResponse("widget.html", {"request": request})
