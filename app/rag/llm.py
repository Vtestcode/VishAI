"""
LLM interaction layer.

Takes retrieved context chunks and the user's question, builds a prompt,
and calls OpenAI to generate a grounded answer.
"""

from __future__ import annotations

import logging
import json
import re
from typing import Iterable, List, Tuple

from langchain_core.documents import Document
from openai import OpenAI

from app.core.cache import build_cache_key, cache_get_json, cache_set_json
from app.core.config import Settings, get_settings
from app.models.schemas import ToolCall, ToolDefinition
from app.rag.mcp import answer_with_mcp, mcp_enabled, route_query_to_tool

logger = logging.getLogger(__name__)

SMALL_TALK_PATTERN = re.compile(
    r"^\s*(hi|hello|hey|yo|how are you|good morning|good afternoon|good evening)\s*[!.?]*\s*$",
    re.IGNORECASE,
)

INSUFFICIENT_INFO_PHRASE = "I don't have enough information to answer that based on the available documents."

SYSTEM_PROMPT = """\
You are a recruiter-facing portfolio assistant.

Your job is to answer questions about Vishal using the retrieved document context.

Rules:
1. Prioritize being helpful, clear, and professional.
2. Answer using the provided context. Summarize and synthesize naturally instead of sounding robotic.
3. If the user asks a vague question like "what is this document about?", infer that they mean the most relevant retrieved document and summarize it.
4. If the user asks which projects stand out, which projects are most impressive, or similar ranking/summarization questions, identify the strongest projects mentioned in the context and briefly explain why they stand out.
5. Do not list or mention sources unless the user explicitly asks for them.
6. Refer to Vishal by name whenever it fits naturally. Do not say "the portfolio owner."
7. If the context still does not contain enough information, respond with exactly this sentence:
"I don't have enough information to answer that based on the available documents."
"""

USER_PROMPT_TEMPLATE = """\
Context:
{context}

Question: {question}
"""


def _format_context(chunks: List[Tuple[Document, float]]) -> str:
    """Join retrieved chunks into a single context string."""
    parts: list[str] = []
    for i, (doc, _score) in enumerate(chunks, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        header = f"[Source {i}: {source}"
        if page != "":
            header += f", page {page}"
        header += "]"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def is_small_talk(message: str) -> bool:
    """Return True for simple greetings that should not require retrieval."""
    return bool(SMALL_TALK_PATTERN.match(message))


def generate_answer(
    question: str,
    chunks: List[Tuple[Document, float]],
    settings: Settings | None = None,
) -> tuple[str, list[ToolDefinition], list[ToolCall], str | None]:
    """Call OpenAI with the retrieved context and return the assistant's answer."""
    if settings is None:
        settings = get_settings()

    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file or environment."
        )

    context_str = _format_context(chunks)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        context=context_str,
        question=question,
    )

    if mcp_enabled(settings):
        routed_tool = route_query_to_tool(question, settings)
        answer, available_tools, tool_calls = answer_with_mcp(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            settings=settings,
            preferred_tool=routed_tool,
        )
        if not answer:
            answer = INSUFFICIENT_INFO_PHRASE

        if not tool_calls:
            answer = validate_answer(question, answer, chunks, settings)
            answer = _apply_contact_fallback(answer, settings)

        logger.info(
            "LLM generated %d-char MCP answer for: %.80s...",
            len(answer),
            question,
        )
        return answer, available_tools, tool_calls, routed_tool

    client = OpenAI(api_key=settings.openai_api_key)

    response = client.chat.completions.create(
        model=settings.model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    answer = (response.choices[0].message.content or "").strip()
    answer = validate_answer(question, answer, chunks, settings)
    answer = _apply_contact_fallback(answer, settings)
    logger.info("LLM generated %d-char answer for: %.80s...", len(answer), question)
    return answer, [], [], None


def generate_tool_only_answer(
    question: str,
    settings: Settings,
    routed_tool: str,
) -> tuple[str, list[ToolDefinition], list[ToolCall], str | None]:
    """Answer with MCP directly when the query should bypass RAG retrieval."""
    if not mcp_enabled(settings):
        raise RuntimeError("MCP is not enabled.")

    cache_key = build_cache_key("tool-answer", f"{routed_tool}|{question}")
    cached = cache_get_json(cache_key)
    if isinstance(cached, dict):
        return (
            str(cached.get("answer", "")),
            [ToolDefinition(**item) for item in cached.get("available_tools", [])],
            [ToolCall(**item) for item in cached.get("tool_calls", [])],
            str(cached.get("routed_tool") or routed_tool),
        )

    answer, available_tools, tool_calls = answer_with_mcp(
        system_prompt=(
            "You are a recruiter-facing assistant with access to external tools. "
            "Use the routed tool when it is relevant to answer the question directly and concisely."
        ),
        user_prompt=question,
        settings=settings,
        preferred_tool=routed_tool,
    )
    if not answer:
        answer = "I couldn't complete that tool request."

    logger.info(
        "LLM generated %d-char tool-only answer via %s for: %.80s...",
        len(answer),
        routed_tool,
        question,
    )
    cache_set_json(
        cache_key,
        {
            "answer": answer,
            "available_tools": [tool.model_dump() for tool in available_tools],
            "tool_calls": [tool_call.model_dump() for tool_call in tool_calls],
            "routed_tool": routed_tool,
        },
        settings.tool_answer_cache_ttl_seconds,
    )
    return answer, available_tools, tool_calls, routed_tool


def stream_answer(
    question: str,
    chunks: List[Tuple[Document, float]],
    settings: Settings | None = None,
) -> Iterable[str]:
    """Stream an answer token by token without applying final validation."""
    if settings is None:
        settings = get_settings()

    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file or environment."
        )

    context_str = _format_context(chunks)
    client = OpenAI(api_key=settings.openai_api_key)

    stream = client.chat.completions.create(
        model=settings.model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    context=context_str,
                    question=question,
                ),
            },
        ],
        temperature=0.2,
        max_tokens=1024,
        stream=True,
    )

    for event in stream:
        token = event.choices[0].delta.content
        if token:
            yield token


def validate_answer(
    question: str,
    answer: str,
    chunks: List[Tuple[Document, float]],
    settings: Settings,
) -> str:
    """Check that the answer is grounded and actually answers the user."""
    if not settings.enable_answer_validation or INSUFFICIENT_INFO_PHRASE in answer:
        return answer

    context_str = _format_context(chunks)
    client = OpenAI(api_key=settings.openai_api_key)

    try:
        response = client.chat.completions.create(
            model=settings.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict RAG answer validator. Decide whether the answer "
                        "is grounded in the context and answers the user's question. Return "
                        "only JSON with keys: grounded, answers_question, corrected_answer. "
                        "If the answer is not grounded or does not answer the question, put "
                        f"this exact fallback in corrected_answer: {INSUFFICIENT_INFO_PHRASE}"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "context": context_str[:12000],
                            "question": question,
                            "answer": answer,
                        },
                        ensure_ascii=True,
                    ),
                },
            ],
            temperature=0,
            max_tokens=500,
        )
        payload = json.loads(response.choices[0].message.content or "{}")
    except Exception as exc:
        logger.info("Answer validation failed; using original answer: %s", exc)
        return answer

    grounded = bool(payload.get("grounded"))
    answers_question = bool(payload.get("answers_question"))
    corrected = str(payload.get("corrected_answer", "")).strip()

    if grounded and answers_question:
        return corrected or answer

    return corrected or INSUFFICIENT_INFO_PHRASE


def finalize_streamed_answer(
    question: str,
    answer: str,
    chunks: List[Tuple[Document, float]],
    settings: Settings,
) -> str:
    """Apply the same validation and contact fallback used by non-streaming chat."""
    answer = validate_answer(question, answer, chunks, settings)
    return _apply_contact_fallback(answer, settings)


def generate_small_talk_answer(
    question: str,
    settings: Settings | None = None,
) -> str:
    """Generate a short conversational reply without document retrieval."""
    if settings is None:
        settings = get_settings()

    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file or environment."
        )

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a friendly portfolio assistant. "
                    "Reply naturally and briefly to greetings or simple pleasantries. "
                    "Invite the user to ask about Vishal when appropriate. "
                    "Do not mention documents or sources unless the user asks."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.4,
        max_tokens=80,
    )

    answer = (response.choices[0].message.content or "").strip()
    logger.info("LLM generated small-talk answer for: %.80s...", question)
    return answer


def stream_small_talk_answer(
    question: str,
    settings: Settings | None = None,
) -> Iterable[str]:
    """Stream a small-talk response token by token."""
    if settings is None:
        settings = get_settings()

    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file or environment."
        )

    client = OpenAI(api_key=settings.openai_api_key)
    stream = client.chat.completions.create(
        model=settings.model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a friendly portfolio assistant. "
                    "Reply naturally and briefly to greetings or simple pleasantries. "
                    "Invite the user to ask about Vishal when appropriate. "
                    "Do not mention documents or sources unless the user asks."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.4,
        max_tokens=80,
        stream=True,
    )

    for event in stream:
        token = event.choices[0].delta.content
        if token:
            yield token


def _apply_contact_fallback(answer: str, settings: Settings) -> str:
    """Redirect to email when the documents do not contain a sufficient answer."""
    if INSUFFICIENT_INFO_PHRASE not in answer:
        return answer

    if settings.contact_email:
        return (
            "I don't have enough information to answer that from the portfolio documents. "
            f"For more details, please email {settings.contact_email}."
        )

    return (
        "I don't have enough information to answer that from the portfolio documents. "
        "Please reach out by email for more details."
    )
