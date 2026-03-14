"""
Chat endpoint.

Accepts a user message, retrieves relevant context when needed,
generates an LLM answer, and returns it to the client.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import Settings, get_settings
from app.models.schemas import ChatRequest, ChatResponse
from app.rag.llm import generate_answer, generate_small_talk_answer, is_small_talk
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
    """
    try:
        if is_small_talk(body.message):
            answer = generate_small_talk_answer(body.message, settings=settings)
            return ChatResponse(answer=answer, sources=[])

        chunks = retrieve_relevant_chunks(body.message, settings=settings)
        answer = generate_answer(body.message, chunks, settings=settings)
        return ChatResponse(answer=answer, sources=[])

    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {exc}",
        )
