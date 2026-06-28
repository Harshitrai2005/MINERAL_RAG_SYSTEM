"""
Metrics Route — /api/metrics
Serves Prometheus text-format scrape endpoint (no auth required by convention,
but you can add API-key gating if your Prometheus instance is public-facing).
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from metrics.prometheus_metrics import generate_metrics_text, vector_store_documents
from core.config import settings

router = APIRouter()


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    summary="Prometheus metrics scrape endpoint",
    include_in_schema=True,
)
async def prometheus_metrics(request: Request):
    """
    Prometheus-compatible text exposition.
    Point your prometheus.yml scrape config at /api/metrics.
    """
    # Refresh vector store document counts on every scrape
    try:
        container = request.app.state.container
        for col_name in [settings.COLLECTION_GEOLOGICAL, settings.COLLECTION_MINERAL]:
            count = container.vector_repo.count(col_name)
            vector_store_documents.labels(collection=col_name).set(count)
    except Exception:
        pass  # Don't fail scrapes due to vector store blips

    return PlainTextResponse(
        content=generate_metrics_text(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
