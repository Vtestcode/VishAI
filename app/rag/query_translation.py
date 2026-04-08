"""Query translation helpers for retrieval."""

from __future__ import annotations

import json
import logging

from openai import OpenAI

from app.core.config import Settings

logger = logging.getLogger(__name__)


def translate_query(query: str, settings: Settings) -> list[str]:
    """Generate retrieval queries that preserve the user's intent."""
    max_queries = max(settings.query_rewrite_count, 1)
    variants = _heuristic_query_variants(query)
    if not settings.openai_api_key or max_queries <= 1:
        return variants[:max_queries]

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite recruiter questions into search queries for a portfolio RAG system. "
                        "Return only JSON with a 'queries' array. Preserve all names, dates, tools, "
                        "and constraints. Include the original query as the first item."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"query": query, "max_queries": max_queries},
                        ensure_ascii=True,
                    ),
                },
            ],
            temperature=0,
            max_tokens=220,
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        llm_queries = [
            str(item).strip()
            for item in payload.get("queries", [])
            if str(item).strip()
        ]
        variants = _dedupe([query, *llm_queries, *_heuristic_query_variants(query)])
    except Exception as exc:
        logger.info("Query translation failed; using heuristic variants: %s", exc)

    return variants[:max_queries]


def _heuristic_query_variants(query: str) -> list[str]:
    """Generate targeted fallback rewrites for recruiter-style questions."""
    normalized = " ".join(query.lower().split())
    variants = [query]

    if any(phrase in normalized for phrase in ["client", "clients", "worked for", "work for"]):
        variants.extend(
            [
                f"{query} employers companies organizations clients consulting engagements",
                "Vishal clients employers companies organizations worked for",
                "consulting clients customer accounts employers organizations Vishal",
            ]
        )

    if any(
        phrase in normalized
        for phrase in [
            "project",
            "projects",
            "stand out",
            "standout",
            "most impressive",
            "best project",
            "best projects",
            "notable work",
        ]
    ):
        variants.extend(
            [
                f"{query} portfolio projects case studies implementations",
                "Vishal notable projects portfolio work achievements",
                "Vishal strongest projects most impressive work notable achievements case studies",
            ]
        )

    if any(phrase in normalized for phrase in ["skill", "skills", "technology", "technologies", "stack"]):
        variants.extend(
            [
                f"{query} technical skills tools frameworks languages cloud data ai",
                "Vishal technical skills technologies stack experience",
            ]
        )

    if any(phrase in normalized for phrase in ["experience", "background", "about"]):
        variants.extend(
            [
                f"{query} resume summary background experience profile",
                "Vishal resume background experience summary",
            ]
        )

    return _dedupe(variants)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = " ".join(value.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped
