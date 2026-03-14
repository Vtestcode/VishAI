"""
Retriever: loads the configured vector store and performs similarity search.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from langchain_core.documents import Document

from app.core.config import Settings, get_settings
from app.rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def retrieve_relevant_chunks(
    query: str,
    settings: Settings | None = None,
    top_k: int | None = None,
) -> List[Tuple[Document, float]]:
    """
    Return the *top_k* most relevant chunks for *query*.

    Each item is a (Document, score) tuple. Lower score means more similar.
    """
    if settings is None:
        settings = get_settings()

    k = top_k or settings.top_k
    store = get_vector_store(settings)

    merged_results: list[tuple[Document, float]] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for expanded_query in _expand_query_variants(query):
        results = store.similarity_search_with_score(expanded_query, k=k)
        for doc, score in results:
            key = (
                str(doc.metadata.get("source", "")),
                str(doc.metadata.get("page", "")),
                str(doc.metadata.get("start_index", "")),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged_results.append((doc, score))

    merged_results.sort(key=lambda item: item[1])
    final_results = merged_results[:k]
    logger.info(
        "Retrieved %d chunk(s) for query: %.80s... using %d query variant(s)",
        len(final_results),
        query,
        len(_expand_query_variants(query)),
    )
    return final_results


def _expand_query_variants(query: str) -> List[str]:
    """Generate a few targeted query rewrites for recruiter-style questions."""
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

    if any(phrase in normalized for phrase in ["project", "projects"]):
        variants.extend(
            [
                f"{query} portfolio projects case studies implementations",
                "Vishal notable projects portfolio work achievements",
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

    # Preserve order while removing duplicates.
    seen: set[str] = set()
    deduped: list[str] = []
    for variant in variants:
        if variant not in seen:
            seen.add(variant)
            deduped.append(variant)
    return deduped
