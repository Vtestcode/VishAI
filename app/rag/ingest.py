"""
Document ingestion pipeline.

Loads text, PDF, and Markdown files from S3, parses useful metadata, creates
section-aware chunks plus RAPTOR-style summaries, and persists them to Pinecone.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
from collections import defaultdict
from typing import Iterable, List

import boto3
from botocore.exceptions import ClientError
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pypdf import PdfReader

from app.core.config import Settings, get_settings
from app.rag.vector_store import rebuild_vector_store, update_vector_store

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf")
MANIFEST_VERSION = 1


def _load_s3_documents(settings: Settings) -> tuple[List[Document], dict[str, dict]]:
    """Load supported documents from S3 and return documents plus source manifest."""
    if not settings.s3_bucket:
        raise RuntimeError(
            "S3_BUCKET is not set. Add it to your environment or .env file."
        )

    s3_client = _get_s3_client(settings)
    paginator = s3_client.get_paginator("list_objects_v2")

    docs: List[Document] = []
    manifest: dict[str, dict] = {}
    prefix = settings.s3_prefix or ""

    for page in paginator.paginate(Bucket=settings.s3_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            lowered_key = key.lower()

            if lowered_key.endswith("/") or not lowered_key.endswith(SUPPORTED_EXTENSIONS):
                if not lowered_key.endswith("/"):
                    logger.info("Skipping unsupported S3 object: s3://%s/%s", settings.s3_bucket, key)
                continue

            response = s3_client.get_object(Bucket=settings.s3_bucket, Key=key)
            content_bytes = response["Body"].read()
            content_sha = hashlib.sha256(content_bytes).hexdigest()
            source = f"s3://{settings.s3_bucket}/{key}"
            file_type = lowered_key.rsplit(".", 1)[-1]
            last_modified = obj.get("LastModified")

            metadata = {
                "source": source,
                "s3_key": key,
                "doc_id": _stable_hash(source),
                "file_name": key.rsplit("/", 1)[-1],
                "file_type": file_type,
                "s3_etag": str(obj.get("ETag", "")).strip('"'),
                "s3_size": int(obj.get("Size", len(content_bytes))),
                "s3_last_modified": last_modified.isoformat() if last_modified else "",
                "content_sha256": content_sha,
            }
            manifest[source] = dict(metadata)

            if lowered_key.endswith(".pdf"):
                docs.extend(_load_pdf_from_bytes(content_bytes, metadata))
            else:
                text = content_bytes.decode("utf-8", errors="replace")
                docs.extend(_load_text_sections(text, metadata))

    logger.info(
        "Loaded %d parsed document section(s) from S3 bucket %s with prefix %s",
        len(docs),
        settings.s3_bucket,
        prefix or "/",
    )
    return docs, manifest


def _load_pdf_from_bytes(content_bytes: bytes, metadata: dict) -> List[Document]:
    """Extract page text from a PDF stored in memory."""
    reader = PdfReader(io.BytesIO(content_bytes))
    docs: List[Document] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={**metadata, "page": page_number, "section": f"page {page_number}"},
                )
            )

    return docs


def _load_text_sections(text: str, metadata: dict) -> List[Document]:
    """Split Markdown/text into heading-aware sections before chunking."""
    sections: list[Document] = []
    current_heading = "document"
    current_lines: list[str] = []

    for line in text.splitlines():
        heading = _extract_heading(line)
        if heading and current_lines:
            sections.append(_section_document(current_lines, metadata, current_heading))
            current_lines = []
        if heading:
            current_heading = heading
        current_lines.append(line)

    if current_lines:
        sections.append(_section_document(current_lines, metadata, current_heading))

    return [section for section in sections if section.page_content.strip()]


def _section_document(lines: list[str], metadata: dict, heading: str) -> Document:
    return Document(
        page_content="\n".join(lines).strip(),
        metadata={**metadata, "section": heading},
    )


def _extract_heading(line: str) -> str:
    markdown_match = re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
    if markdown_match:
        return markdown_match.group(1).strip()

    stripped = line.strip()
    if stripped and len(stripped) <= 80 and stripped.endswith(":"):
        return stripped[:-1]

    return ""


def _split_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Split documents on semantic-ish boundaries: headings, paragraphs, sentences."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        source = str(chunk.metadata.get("source", ""))
        page = str(chunk.metadata.get("page", ""))
        start_index = str(chunk.metadata.get("start_index", i))
        content_hash = _stable_hash(chunk.page_content)
        chunk.metadata.update(
            {
                "chunk_type": "leaf",
                "chunk_index": i,
                "content_hash": content_hash,
                "chunk_id": _stable_hash("|".join([source, page, start_index, content_hash])),
            }
        )

    logger.info("Split into %d semantic chunk(s)", len(chunks))
    return chunks


def _build_raptor_summary_chunks(
    chunks: List[Document],
    settings: Settings,
) -> List[Document]:
    """Create lightweight RAPTOR-style parent summaries for groups of leaf chunks."""
    if not settings.enable_raptor:
        return []

    client = OpenAI(api_key=settings.openai_api_key)
    summary_docs: list[Document] = []
    chunks_by_source: dict[str, list[Document]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_source[str(chunk.metadata.get("source", ""))].append(chunk)

    for source, source_chunks in chunks_by_source.items():
        if len(source_chunks) < 2:
            continue

        for group_index, group in enumerate(_batched(source_chunks, settings.raptor_group_size)):
            summary = _summarize_chunk_group(client, settings, group)
            if not summary:
                continue

            base_metadata = dict(group[0].metadata)
            child_ids = [str(doc.metadata.get("chunk_id", "")) for doc in group]
            summary_id = _stable_hash(f"raptor|{source}|{group_index}|{'|'.join(child_ids)}")
            summary_docs.append(
                Document(
                    page_content=summary,
                    metadata={
                        **base_metadata,
                        "chunk_type": "raptor_summary",
                        "raptor_level": 1,
                        "raptor_group": group_index,
                        "child_chunk_ids": ",".join(child_ids),
                        "chunk_id": summary_id,
                        "content_hash": _stable_hash(summary),
                        "start_index": "",
                    },
                )
            )

    logger.info("Created %d RAPTOR summary chunk(s)", len(summary_docs))
    return summary_docs


def _summarize_chunk_group(client: OpenAI, settings: Settings, chunks: list[Document]) -> str:
    text = "\n\n---\n\n".join(chunk.page_content[:1800] for chunk in chunks)
    response = client.chat.completions.create(
        model=settings.model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "Summarize this portfolio knowledge-base context for retrieval. "
                    "Keep specific names, projects, skills, companies, tools, outcomes, and dates. "
                    "Do not add facts not present in the context."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_tokens=260,
    )
    return (response.choices[0].message.content or "").strip()


def _build_vector_store(
    chunks: List[Document],
    settings: Settings,
    deleted_sources: Iterable[str] = (),
    rebuild: bool = False,
):
    """Create, rebuild, or incrementally update Pinecone from chunks."""
    if rebuild:
        vector_store = rebuild_vector_store(settings, chunks)
    else:
        vector_store = update_vector_store(settings, chunks, deleted_sources=deleted_sources)
    logger.info("Persisted %d vector(s) to Pinecone", len(chunks))
    return vector_store


def run_ingest(settings: Settings | None = None, rebuild: bool = False) -> int:
    """
    Ingest S3 knowledge-base files into Pinecone.

    1. Load and parse supported files from S3.
    2. Compare against the S3-backed manifest for incremental indexing.
    3. Split changed files into section-aware chunks and RAPTOR summaries.
    4. Embed and persist changed vectors to Pinecone.
    """
    if settings is None:
        settings = get_settings()

    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file or environment."
        )

    s3_client = _get_s3_client(settings)
    existing_manifest = {} if rebuild else _read_manifest(s3_client, settings)
    docs, next_manifest = _load_s3_documents(settings)
    deleted_sources = sorted(set(existing_manifest) - set(next_manifest))

    if rebuild:
        docs_to_index = docs
    else:
        docs_to_index = [
            doc for doc in docs if _source_changed(doc.metadata, existing_manifest)
        ]

    if not docs and rebuild:
        logger.warning("No documents found in S3; vector store will be empty.")
        rebuild_vector_store(settings, [])
        _write_manifest(s3_client, settings, next_manifest)
        return 0

    if not docs_to_index and not deleted_sources:
        logger.info("No S3 knowledge-base changes detected; skipping Pinecone update.")
        return 0

    leaf_chunks = _split_documents(
        docs_to_index,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    summary_chunks = _build_raptor_summary_chunks(leaf_chunks, settings)
    chunks = leaf_chunks + summary_chunks

    _build_vector_store(chunks, settings=settings, deleted_sources=deleted_sources, rebuild=rebuild)
    _write_manifest(s3_client, settings, next_manifest)
    return len(chunks)


def _source_changed(metadata: dict, existing_manifest: dict[str, dict]) -> bool:
    source = str(metadata.get("source", ""))
    previous = existing_manifest.get(source)
    if not previous:
        return True
    return previous.get("content_sha256") != metadata.get("content_sha256")


def _read_manifest(s3_client, settings: Settings) -> dict[str, dict]:
    key = _manifest_key(settings)
    try:
        response = s3_client.get_object(Bucket=settings.s3_bucket, Key=key)
        payload = json.loads(response["Body"].read().decode("utf-8"))
        return payload.get("sources", {})
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code in {"NoSuchKey", "404"}:
            return {}
        raise


def _write_manifest(s3_client, settings: Settings, sources: dict[str, dict]) -> None:
    key = _manifest_key(settings)
    payload = {
        "version": MANIFEST_VERSION,
        "pinecone_index_name": settings.pinecone_index_name,
        "pinecone_namespace": settings.pinecone_namespace,
        "sources": sources,
    }
    s3_client.put_object(
        Bucket=settings.s3_bucket,
        Key=key,
        Body=json.dumps(payload, ensure_ascii=True, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def _manifest_key(settings: Settings) -> str:
    if settings.rag_manifest_key:
        return settings.rag_manifest_key
    prefix = settings.s3_prefix.strip("/")
    if prefix:
        return f"{prefix}/.rag-index-manifest.json"
    return ".rag-index-manifest.json"


def _get_s3_client(settings: Settings):
    client_kwargs = {}
    if settings.aws_region:
        client_kwargs["region_name"] = settings.aws_region
    return boto3.client("s3", **client_kwargs)


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _batched(items: list[Document], size: int) -> Iterable[list[Document]]:
    size = max(size, 1)
    for i in range(0, len(items), size):
        yield items[i : i + size]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    count = run_ingest()
    print(f"Ingestion complete - {count} chunk(s) stored.")
