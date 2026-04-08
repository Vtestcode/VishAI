"""LLM reranking for retrieved chunks."""

from __future__ import annotations

import json
import logging
from typing import List, Tuple

from langchain_core.documents import Document
from openai import OpenAI

from app.core.config import Settings

logger = logging.getLogger(__name__)


def rerank_chunks(
    query: str,
    chunks: List[Tuple[Document, float]],
    settings: Settings,
    top_k: int,
) -> List[Tuple[Document, float]]:
    """Use the chat model as a cross-encoder-style reranker over retrieved candidates."""
    if not settings.enable_reranking or len(chunks) <= top_k:
        return chunks[:top_k]

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        candidates = [
            {
                "id": i,
                "source": doc.metadata.get("source", ""),
                "section": doc.metadata.get("section", ""),
                "chunk_type": doc.metadata.get("chunk_type", ""),
                "text": doc.page_content[:1200],
            }
            for i, (doc, _score) in enumerate(chunks)
        ]
        response = client.chat.completions.create(
            model=settings.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rerank retrieved portfolio context chunks for answering the user. "
                        "Prefer chunks that directly answer the question and are specific. "
                        "Return only JSON with an 'ids' array ordered best to worst."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"query": query, "top_k": top_k, "candidates": candidates},
                        ensure_ascii=True,
                    ),
                },
            ],
            temperature=0,
            max_tokens=220,
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        ranked_ids = [int(item) for item in payload.get("ids", [])]
    except Exception as exc:
        logger.info("Reranking failed; using vector order: %s", exc)
        return chunks[:top_k]

    by_id = {i: item for i, item in enumerate(chunks)}
    reranked: list[tuple[Document, float]] = []
    for item_id in ranked_ids:
        if item_id in by_id and by_id[item_id] not in reranked:
            reranked.append(by_id[item_id])

    for item in chunks:
        if len(reranked) >= top_k:
            break
        if item not in reranked:
            reranked.append(item)

    return reranked[:top_k]
