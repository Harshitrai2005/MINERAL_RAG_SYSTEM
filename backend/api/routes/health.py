"""Health-check routes."""
from __future__ import annotations

from fastapi import APIRouter, Request
from core.config import settings

router = APIRouter()


@router.get("/config", summary="Public runtime config for frontend")
async def public_config():
    """Returns only the gateway API_KEY so the frontend never has it hardcoded.
    GROQ_API_KEY and QDRANT_API_KEY are never exposed here."""
    return {"api_key": settings.API_KEY}


@router.get("/health", summary="System health check")
async def health_check(request: Request):
    container = request.app.state.container
    vector_ok = container.vector_repo.health_check()
    geo_count  = container.vector_repo.count(settings.COLLECTION_GEOLOGICAL)
    min_count  = container.vector_repo.count(settings.COLLECTION_MINERAL)

    return {
        "status": "healthy" if vector_ok else "degraded",
        "version": settings.VERSION,
        "vector_backend": settings.VECTOR_BACKEND,
        "vector_store": "ok" if vector_ok else "error",
        "collections": {
            settings.COLLECTION_GEOLOGICAL: geo_count,
            settings.COLLECTION_MINERAL:    min_count,
        },
        "total_chunks": geo_count + min_count,
    }