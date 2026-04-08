"""
Retriever: loads the configured vector store and performs similarity search.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from langchain_core.documents import Document

from app.core.config import Settings, get_settings
from app.rag.query_translation import translate_query
from app.rag.reranker import rerank_chunks
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
    candidate_k = max(settings.retrieval_candidate_k, k)
    store = get_vector_store(settings)

    merged_results: list[tuple[Document, float]] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    expanded_queries = translate_query(query, settings)

    for expanded_query in expanded_queries:
        results = store.similarity_search_with_score(expanded_query, k=candidate_k)
        for doc, score in results:
            key = (
                str(doc.metadata.get("source", "")),
                str(doc.metadata.get("page", "")),
                str(doc.metadata.get("start_index", "")),
                str(doc.metadata.get("chunk_id", "")),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged_results.append((doc, score))

    merged_results.sort(key=lambda item: item[1])
    final_results = rerank_chunks(query, merged_results, settings, top_k=k)
    logger.info(
        "Retrieved %d reranked chunk(s) for query: %.80s... using %d query variant(s)",
        len(final_results),
        query,
        len(expanded_queries),
    )
    return final_results
