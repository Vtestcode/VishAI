"""
Document ingestion endpoint.

Re-ingests every supported file in S3 and rebuilds Pinecone from scratch.
"""

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import Settings, get_settings
from app.models.schemas import IngestResponse
from app.rag.ingest import run_ingest

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    """Rebuild Pinecone from all supported files in the configured S3 bucket."""
    try:
        count = run_ingest(settings)
        return IngestResponse(
            status="ok",
            documents_ingested=count,
            message=f"Successfully ingested {count} chunk(s).",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {exc}",
        )
