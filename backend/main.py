"""
Mineral Exploration Intelligence System
Main FastAPI Application Entry Point — v4.0
Added: Prometheus metrics, Grafana-ready observability
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routes import query, ingest, analysis, health
from api.routes.metrics import router as metrics_router
from core.config import settings
from core.container import build_container
from core.security import validate_groq_api_key, RateLimitMiddleware, query_rate_limiter, ingest_rate_limiter
from metrics.middleware import PrometheusMiddleware
from services.ingestion_queue import IngestionQueue
from utils.logger import setup_logger

logger = setup_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        f"Starting {settings.APP_NAME} v{settings.VERSION} "
        f"[backend={settings.VECTOR_BACKEND}, llm={settings.LLM_PROVIDER}, "
        f"reranker={settings.RERANKER}]"
    )

    logger.info("Validating critical configuration...")

    if (settings.LLM_PROVIDER or "groq").lower() == "groq":
        try:
            validate_groq_api_key(settings.GROQ_API_KEY)
        except RuntimeError as e:
            logger.critical(f"[FATAL] {e}")
            raise

    if not settings.API_KEY:
        logger.critical("[FATAL] API_KEY not set. Set it in .env file.")
        raise RuntimeError("API_KEY is required")

    logger.info("[OK] API_KEY validated")

    app.state.container = build_container(settings)

    ingestion_queue = IngestionQueue(
        ingestion_service=app.state.container.ingestion_service
    )
    app.state.ingestion_queue = ingestion_queue
    asyncio.create_task(ingestion_queue.worker())
    logger.info("[OK] Async ingestion queue worker started")
    logger.info("[OK] Prometheus metrics enabled at /api/metrics")
    logger.info("System ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "RAG-powered mineral exploration intelligence system. "
        "Layered architecture: API → Service → Repository → Infra. "
        "Swappable vector store (LanceDB/Qdrant) and LLM (Groq/OpenAI). "
        "Prometheus metrics at /api/metrics."
    ),
    version=settings.VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

# ── Prometheus middleware (FIRST — wraps everything) ──────────────────────────
app.add_middleware(PrometheusMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rate Limiting ─────────────────────────────────────────────────────────────
app.add_middleware(
    RateLimitMiddleware,
    rate_limiter=ingest_rate_limiter,
    paths=["/api/ingest"],
)
app.add_middleware(
    RateLimitMiddleware,
    rate_limiter=query_rate_limiter,
    paths=["/api/query", "/api/analysis"],
)

# ── Routes ───────────────────────────────────────────────────────────────────
app.include_router(health.router,    prefix="/api",          tags=["Health"])
app.include_router(metrics_router,   prefix="/api",          tags=["Observability"])
app.include_router(ingest.router,    prefix="/api/ingest",   tags=["Data Ingestion"])
app.include_router(query.router,     prefix="/api/query",    tags=["Query & RAG"])
app.include_router(analysis.router,  prefix="/api/analysis", tags=["Mineral Analysis"])

static_dir = FRONTEND_DIR / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
