"""
Document ingestion pipeline.

Loads text, PDF, and Markdown files from S3, splits them into chunks,
computes embeddings, and persists them to Pinecone.
"""

from __future__ import annotations

import io
import logging
from typing import List

import boto3
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.core.config import Settings, get_settings
from app.rag.vector_store import rebuild_vector_store

logger = logging.getLogger(__name__)


def _load_s3_documents(settings: Settings) -> List[Document]:
    """Load supported documents from an S3 bucket."""
    if not settings.s3_bucket:
        raise RuntimeError(
            "S3_BUCKET is not set. Add it to your environment or .env file."
        )

    client_kwargs = {}
    if settings.aws_region:
        client_kwargs["region_name"] = settings.aws_region

    s3_client = boto3.client("s3", **client_kwargs)
    paginator = s3_client.get_paginator("list_objects_v2")

    docs: List[Document] = []
    prefix = settings.s3_prefix or ""

    for page in paginator.paginate(Bucket=settings.s3_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            lowered_key = key.lower()

            if lowered_key.endswith("/"):
                continue

            if not lowered_key.endswith((".txt", ".md", ".pdf")):
                logger.info(
                    "Skipping unsupported S3 object: s3://%s/%s",
                    settings.s3_bucket,
                    key,
                )
                continue

            response = s3_client.get_object(Bucket=settings.s3_bucket, Key=key)
            content_bytes = response["Body"].read()
            source = f"s3://{settings.s3_bucket}/{key}"

            if lowered_key.endswith(".pdf"):
                docs.extend(_load_pdf_from_bytes(content_bytes, source))
            else:
                docs.append(
                    Document(
                        page_content=content_bytes.decode("utf-8", errors="replace"),
                        metadata={"source": source},
                    )
                )

    logger.info(
        "Loaded %d raw document(s) from S3 bucket %s with prefix %s",
        len(docs),
        settings.s3_bucket,
        prefix or "/",
    )
    return docs


def _load_pdf_from_bytes(content_bytes: bytes, source: str) -> List[Document]:
    """Extract page text from a PDF stored in memory."""
    reader = PdfReader(io.BytesIO(content_bytes))
    docs: List[Document] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": source, "page": page_number},
                )
            )

    return docs


def _split_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Split documents into smaller chunks suitable for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split into %d chunk(s)", len(chunks))
    return chunks


def _build_vector_store(chunks: List[Document], settings: Settings):
    """Create or rebuild Pinecone from *chunks*."""
    vector_store = rebuild_vector_store(settings, chunks)
    logger.info("Persisted %d vectors to Pinecone", len(chunks))
    return vector_store


def run_ingest(settings: Settings | None = None) -> int:
    """
    Full ingestion pipeline. Returns the number of chunks stored.

    1. Load documents from S3.
    2. Split them into chunks.
    3. Embed and persist to Pinecone.
    """
    if settings is None:
        settings = get_settings()

    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file or environment."
        )

    docs = _load_s3_documents(settings)
    if not docs:
        logger.warning("No documents found in S3; vector store will be empty.")
        return 0

    chunks = _split_documents(
        docs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    _build_vector_store(chunks, settings=settings)
    return len(chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    count = run_ingest()
    print(f"Ingestion complete - {count} chunk(s) stored.")
