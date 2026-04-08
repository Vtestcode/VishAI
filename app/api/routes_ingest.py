"""
Document ingestion endpoint.

Indexes supported S3 files into Pinecone, either incrementally or by rebuild.
"""

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.config import Settings, get_settings
from app.models.schemas import IngestResponse
from app.rag.ingest import run_ingest

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    rebuild: bool = Query(
        default=False,
        description="Delete and rebuild the entire Pinecone namespace instead of indexing only S3 changes.",
    ),
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    """Index supported files from the configured S3 bucket into Pinecone."""
    try:
        count = run_ingest(settings, rebuild=rebuild)
        mode = "rebuild" if rebuild else "incremental"
        return IngestResponse(
            status="ok",
            documents_ingested=count,
            mode=mode,
            message=f"Successfully completed {mode} ingest with {count} chunk(s) added.",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {exc}",
        )
