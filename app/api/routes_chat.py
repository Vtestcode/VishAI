"""
Chat endpoint.

Accepts a user message, retrieves relevant context when needed,
generates an LLM answer, and persists the conversation in S3.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import boto3
from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.core.config import Settings, get_settings
from app.models.schemas import ChatRequest, ChatResponse, SourceSnippet
from app.rag.llm import (
    finalize_streamed_answer,
    generate_answer,
    generate_small_talk_answer,
    is_small_talk,
    stream_answer,
    stream_small_talk_answer,
)
from app.rag.retriever import retrieve_relevant_chunks

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    """
    Answer a user message.

    Simple greetings are handled conversationally without retrieval.
    Document questions use retrieval-augmented generation.
    All chats are persisted in S3 by session.
    """
    session_id = body.session_id or uuid4().hex

    try:
        if is_small_talk(body.message):
            answer = generate_small_talk_answer(body.message, settings=settings)
            sources = []
        else:
            chunks = retrieve_relevant_chunks(body.message, settings=settings)
            answer = generate_answer(body.message, chunks, settings=settings)
            sources = [
                SourceSnippet(
                    text=doc.page_content[:500],
                    metadata=doc.metadata,
                )
                for doc, _score in chunks
            ]

        _append_chat_message(settings, session_id, "user", body.message)
        _append_chat_message(settings, session_id, "assistant", answer)
        return ChatResponse(answer=answer, sources=sources, session_id=session_id)

    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {exc}",
        )


@router.post("/chat/stream")
async def chat_stream(
    body: ChatRequest,
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """Stream a chat response as newline-delimited JSON events."""
    session_id = body.session_id or uuid4().hex

    def event_stream():
        answer_parts: list[str] = []
        chunks = []

        try:
            yield _json_event("session", {"session_id": session_id})

            if is_small_talk(body.message):
                yield _json_event("status", {"message": "Composing answer..."})
                for token in stream_small_talk_answer(body.message, settings=settings):
                    answer_parts.append(token)
                    yield _json_event("token", {"text": token})
                answer = "".join(answer_parts).strip()
            else:
                yield _json_event("status", {"message": "Searching the knowledge base..."})
                chunks = retrieve_relevant_chunks(body.message, settings=settings)
                sources = [
                    {
                        "text": doc.page_content[:500],
                        "metadata": doc.metadata,
                    }
                    for doc, _score in chunks
                ]
                yield _json_event("sources", {"sources": sources})
                yield _json_event("status", {"message": "Writing the answer..."})
                for token in stream_answer(body.message, chunks, settings=settings):
                    answer_parts.append(token)
                    yield _json_event("token", {"text": token})

                streamed_answer = "".join(answer_parts).strip()
                yield _json_event("status", {"message": "Checking the answer..."})
                answer = finalize_streamed_answer(body.message, streamed_answer, chunks, settings)
                if answer != streamed_answer:
                    yield _json_event("replace", {"text": answer})

            _append_chat_message(settings, session_id, "user", body.message)
            _append_chat_message(settings, session_id, "assistant", answer)
            yield _json_event("done", {"answer": answer, "session_id": session_id})
        except Exception as exc:
            yield _json_event("error", {"message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache"},
    )


def _append_chat_message(
    settings: Settings,
    session_id: str,
    role: str,
    content: str,
) -> None:
    """Append a message to the persisted S3 chat log for a session."""
    bucket = settings.chat_log_bucket or settings.s3_bucket
    if not bucket:
        raise RuntimeError(
            "CHAT_LOG_BUCKET or S3_BUCKET must be set to store chat logs in S3."
        )

    key = _build_chat_log_key(settings, session_id)
    s3_client = _get_s3_client(settings)
    payload = _read_chat_log(s3_client, bucket, key)

    timestamp = datetime.now(timezone.utc).isoformat()
    payload["updated_at"] = timestamp
    payload["messages"].append(
        {
            "role": role,
            "content": content,
            "timestamp": timestamp,
        }
    )

    s3_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, ensure_ascii=True, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def _get_s3_client(settings: Settings):
    client_kwargs = {}
    if settings.aws_region:
        client_kwargs["region_name"] = settings.aws_region
    return boto3.client("s3", **client_kwargs)


def _build_chat_log_key(settings: Settings, session_id: str) -> str:
    prefix = settings.chat_log_prefix.strip()
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return f"{prefix}{session_id}.json"


def _read_chat_log(s3_client, bucket: str, key: str) -> dict:
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return json.loads(response["Body"].read().decode("utf-8"))
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "")
        if error_code not in {"NoSuchKey", "404"}:
            raise

    timestamp = datetime.now(timezone.utc).isoformat()
    session_id = key.rsplit("/", 1)[-1].removesuffix(".json")
    return {
        "session_id": session_id,
        "created_at": timestamp,
        "updated_at": timestamp,
        "messages": [],
    }


def _json_event(event_type: str, payload: dict) -> str:
    return json.dumps({"type": event_type, **payload}, ensure_ascii=True) + "\n"
