"""
LLM interaction layer.

Takes retrieved context chunks and the user's question, builds a prompt,
and calls OpenAI to generate a grounded answer.
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

from langchain_core.documents import Document
from openai import OpenAI

from app.core.config import Settings, get_settings

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
) -> str:
    """Call OpenAI with the retrieved context and return the assistant's answer."""
    if settings is None:
        settings = get_settings()

    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file or environment."
        )

    context_str = _format_context(chunks)
    client = OpenAI(api_key=settings.openai_api_key)

    response = client.chat.completions.create(
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
    )

    answer = (response.choices[0].message.content or "").strip()
    answer = _apply_contact_fallback(answer, settings)
    logger.info("LLM generated %d-char answer for: %.80s...", len(answer), question)
    return answer


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
