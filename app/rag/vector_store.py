"""
Vector store helpers for Pinecone-backed retrieval.
"""

from __future__ import annotations

import logging
import time

from langchain_openai import OpenAIEmbeddings

from app.core.config import Settings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


def get_embeddings(settings: Settings) -> OpenAIEmbeddings:
    """Return the shared embeddings client used by Pinecone."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=settings.openai_api_key,
    )


def get_vector_store(settings: Settings):
    """Return the configured Pinecone vector store instance."""
    return _get_pinecone_vector_store(settings)


def rebuild_vector_store(settings: Settings, documents):
    """Replace the configured Pinecone namespace contents with the supplied documents."""
    from pinecone.openapi_support.exceptions import PineconeException

    vector_store = _get_pinecone_vector_store(settings, ensure_exists=True)
    namespace = _get_pinecone_namespace(settings)

    try:
        vector_store.index.delete(delete_all=True, namespace=namespace)
    except PineconeException as exc:
        if "Namespace not found" not in str(exc):
            raise
        logger.info(
            "Pinecone namespace %s does not exist yet; continuing with first ingest.",
            namespace,
        )

    vector_store.add_documents(documents=documents)
    return vector_store


def _get_pinecone_vector_store(settings: Settings, ensure_exists: bool = False):
    """Build a Pinecone-backed LangChain vector store."""
    if not settings.pinecone_api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. Add it to your environment or .env file."
        )
    if not settings.pinecone_index_name:
        raise RuntimeError(
            "PINECONE_INDEX_NAME is not set. Add it to your environment or .env file."
        )

    try:
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone, ServerlessSpec
    except ImportError as exc:
        raise RuntimeError(
            "Pinecone dependencies are missing. Install requirements.txt first."
        ) from exc

    client = Pinecone(api_key=settings.pinecone_api_key)
    index_name = settings.pinecone_index_name

    if ensure_exists and not client.has_index(index_name):
        logger.info(
            "Creating Pinecone index %s in %s/%s",
            index_name,
            settings.pinecone_cloud,
            settings.pinecone_region,
        )
        client.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )
        _wait_for_pinecone_index(client, index_name)
    elif not client.has_index(index_name):
        raise RuntimeError(
            f"Pinecone index '{index_name}' does not exist yet. "
            "Run /ingest first so the app can create and populate it."
        )

    return PineconeVectorStore(
        index=client.Index(index_name),
        embedding=get_embeddings(settings),
        namespace=_get_pinecone_namespace(settings),
    )


def _get_pinecone_namespace(settings: Settings) -> str:
    return settings.pinecone_namespace or "rag-docs"


def _wait_for_pinecone_index(client, index_name: str, timeout_seconds: int = 60) -> None:
    """Wait for a new Pinecone index to become ready."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if client.describe_index(index_name).status["ready"]:
            return
        time.sleep(2)

    raise RuntimeError(
        f"Pinecone index '{index_name}' was created but did not become ready in time."
    )
