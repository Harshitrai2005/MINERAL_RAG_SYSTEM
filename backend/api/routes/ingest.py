"""
Ingestion Routes
─────────────────────────────────────────────────────────────────────────────
Upload, list, and delete endpoints for PDF reports and structured datasets.

TWO UPLOAD MODES:
  POST /api/ingest/upload         — synchronous (small files, returns result immediately)
  POST /api/ingest/upload-async   — async/queued (large files, returns job_id immediately)
  GET  /api/ingest/job/{job_id}   — poll job status for async uploads
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from core.config import settings
from core.security import validate_api_key, validate_file_mime_type, validate_domain_content
from services.ingestion_service import IngestionService, UnsupportedFileTypeError, NoContentExtractedError
from utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()


# ── Response schemas ────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    success: bool
    file_name: str
    collection: str
    chunks_added: int
    message: str


class AsyncIngestResponse(BaseModel):
    accepted: bool
    job_id: str
    file_name: str
    message: str


class DeleteResponse(BaseModel):
    success: bool
    source_name: str
    collection: str
    chunks_deleted: int
    message: str


# ── Dependencies ───────────────────────────────────────────────────────────

def get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.container.ingestion_service


def get_ingestion_queue(request: Request):
    return request.app.state.ingestion_queue


# ── Shared file-read helper ────────────────────────────────────────────────

async def _read_and_validate_file(
    file: UploadFile,
    api_key: str,
    category: str,
) -> tuple[bytes, Path]:
    """
    Read file bytes, enforce size limit, validate MIME type.
    Returns (content_bytes, dest_path).
    """
    if category not in ("report", "dataset"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category '{category}'. Must be 'report' or 'dataset'.",
        )

    # Read entire file (UploadFile does not support async iteration)
    content = await file.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum upload size of {settings.MAX_UPLOAD_SIZE_MB} MB.",
        )

    is_valid, error_msg = validate_file_mime_type(
        filename=file.filename,
        content_type=file.content_type,
        content=content,
    )
    if not is_valid:
        logger.warning(f"File validation failed for {file.filename}: {error_msg}")
        raise HTTPException(status_code=400, detail=f"Invalid file: {error_msg}")

    # Domain content validation — reject non-geo/mineral documents
    domain_valid, domain_msg = validate_domain_content(file.filename, content, category)
    if not domain_valid:
        logger.warning(f"Domain validation rejected {file.filename}: {domain_msg}")
        raise HTTPException(status_code=422, detail=domain_msg)

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest_path = upload_dir / file.filename
    dest_path.write_bytes(content)

    return content, dest_path


# ── Synchronous upload ──────────────────────────────────────────────────────

@router.post("/upload", response_model=IngestResponse, summary="Upload and index a file (synchronous)")
async def upload_file(
    file: UploadFile = File(...),
    category: str = Form(..., description="'report' for PDFs/TXT, 'dataset' for CSV/JSON"),
    ingestion: IngestionService = Depends(get_ingestion_service),
    api_key: str = Depends(validate_api_key),
):
    """
    Upload a file and index it synchronously. Returns the result immediately.

    Best for files under ~10MB. For larger files, use /upload-async to avoid
    HTTP timeout issues.

    SECURITY: Requires 'X-API-Key' header with valid API key.
    """
    _, dest_path = await _read_and_validate_file(file, api_key, category)

    try:
        result = ingestion.ingest_file(dest_path, category)
        return IngestResponse(**result)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NoContentExtractedError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Ingestion failed for {file.filename}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during ingestion.")


# ── Async upload (background queue) ────────────────────────────────────────

@router.post(
    "/upload-async",
    response_model=AsyncIngestResponse,
    status_code=202,
    summary="Upload and index a file (async — returns immediately)",
)
async def upload_file_async(
    file: UploadFile = File(...),
    category: str = Form(..., description="'report' for PDFs/TXT, 'dataset' for CSV/JSON"),
    queue=Depends(get_ingestion_queue),
    api_key: str = Depends(validate_api_key),
):
    """
    Upload a file and queue it for background indexing. Returns 202 immediately
    with a job_id. Poll GET /api/ingest/job/{job_id} for completion status.

    Use this for large PDFs (10MB+) to avoid HTTP timeout on Render's free tier.

    SECURITY: Requires 'X-API-Key' header with valid API key.
    """
    _, dest_path = await _read_and_validate_file(file, api_key, category)

    job = queue.enqueue(dest_path, category)
    return AsyncIngestResponse(
        accepted=True,
        job_id=job.job_id,
        file_name=file.filename,
        message=f"File accepted and queued for indexing. Poll /api/ingest/job/{job.job_id} for status.",
    )


# ── Job status ─────────────────────────────────────────────────────────────

@router.get("/job/{job_id}", summary="Poll status of an async ingestion job")
async def get_job_status(job_id: str, queue=Depends(get_ingestion_queue)):
    """
    Poll the status of a background ingestion job started via /upload-async.

    Returns one of: queued | processing | done | failed
    """
    job = queue.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job.to_dict()


@router.get("/jobs", summary="List recent ingestion jobs")
async def list_jobs(limit: int = 50, queue=Depends(get_ingestion_queue)):
    """Return the most recent ingestion jobs (newest first)."""
    return {
        "jobs": queue.list_jobs(limit=limit),
        "queue_depth": queue.queue_depth,
    }


# ── List indexed files ──────────────────────────────────────────────────────

@router.get("/files", summary="List all indexed source files")
async def list_files(ingestion: IngestionService = Depends(get_ingestion_service)):
    """
    Return every source file currently indexed across all collections,
    with chunk counts.  Used by the frontend 'Manage Documents' panel.
    """
    files = ingestion.list_indexed_files()
    return {"files": files, "total": len(files)}


# ── Delete a document ───────────────────────────────────────────────────────

@router.delete("/document", response_model=DeleteResponse, summary="Remove a document from the index")
async def delete_document(
    source_name: str,
    category: str,
    ingestion: IngestionService = Depends(get_ingestion_service),
    api_key: str = Depends(validate_api_key),
):
    """
    Delete all indexed chunks for a given source file.

    - **source_name**: exact filename as stored at upload time (e.g. "report.pdf")
    - **category**: 'report' or 'dataset'
    """
    if category not in ("report", "dataset"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category '{category}'. Must be 'report' or 'dataset'.",
        )
    try:
        result = ingestion.delete_file(source_name, category)
        return DeleteResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Delete failed for '{source_name}': {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during deletion.")


# ── Delete all files of a given type ────────────────────────────────────────

class DeleteByTypeResponse(BaseModel):
    success: bool
    file_type: str
    files_removed: list[str]
    files_count: int
    chunks_deleted: int
    message: str


@router.delete(
    "/by-type/{file_type}",
    response_model=DeleteByTypeResponse,
    summary="Delete all indexed files of a given extension (pdf, csv, json, txt)",
)
async def delete_by_file_type(
    file_type: str,
    ingestion: IngestionService = Depends(get_ingestion_service),
    api_key: str = Depends(validate_api_key),
):
    """
    Remove every indexed chunk belonging to files with the given extension.

    - **file_type**: one of `pdf`, `csv`, `json`, `txt`

    Scans all collections and deletes every source file whose name ends with
    `.{file_type}`. Returns the list of files removed and total chunks deleted.

    SECURITY: Requires 'X-API-Key' header.
    """
    allowed = {"pdf", "csv", "json", "txt"}
    ft = file_type.lower().lstrip(".")
    if ft not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ft}'. Allowed: {', '.join(sorted(allowed))}",
        )
    try:
        result = ingestion.delete_by_file_type(ft)
        return DeleteByTypeResponse(**result)
    except Exception as exc:
        logger.error(f"delete_by_file_type failed for '{ft}': {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during bulk delete.")


# ── Stats ───────────────────────────────────────────────────────────────────

@router.get("/stats", summary="Document counts per collection")
async def get_stats(request: Request):
    """Quick summary: total chunks per collection."""
    repo = request.app.state.container.vector_repo
    geo  = repo.count(settings.COLLECTION_GEOLOGICAL)
    min_ = repo.count(settings.COLLECTION_MINERAL)
    return {
        "total_documents": geo + min_,
        "collections": [
            {"collection_name": settings.COLLECTION_GEOLOGICAL, "document_count": geo},
            {"collection_name": settings.COLLECTION_MINERAL,    "document_count": min_},
        ],
    }