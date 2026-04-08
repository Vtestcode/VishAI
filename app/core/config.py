"""
Application configuration loaded from environment variables.

Uses pydantic-settings to read from .env files and environment variables.
All secrets (like OPENAI_API_KEY) are read from the environment — never hard-coded.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the RAG chatbot."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- OpenAI ---
    openai_api_key: str = ""
    contact_email: str = os.getenv("CONTACT_EMAIL", "")

    # --- LLM model ---
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o-mini")

    # --- RAG tuning ---
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    retrieval_candidate_k: int = 15
    query_rewrite_count: int = 3
    enable_reranking: bool = True
    enable_answer_validation: bool = True
    enable_raptor: bool = True
    raptor_group_size: int = 6

    # --- S3 document storage ---
    s3_bucket: str = os.getenv("S3_BUCKET", "")
    s3_prefix: str = os.getenv("S3_PREFIX", "")
    aws_region: str = os.getenv("AWS_REGION", "")
    rag_manifest_key: str = os.getenv("RAG_MANIFEST_KEY", "")
    chat_log_bucket: str = os.getenv("CHAT_LOG_BUCKET", "")
    chat_log_prefix: str = os.getenv("CHAT_LOG_PREFIX", "chat-logs/")

    # --- Pinecone vector store ---
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE", "rag-docs")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")

@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance (created once per process)."""
    return Settings()
