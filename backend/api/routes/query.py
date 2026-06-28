"""
Query Routes
Natural-language query endpoints over the ingested knowledge base.


  - Exposes clarifying_questions and needs_clarification in response.
  - Saves each query result to SQLite metrics DB (persistent across restarts).
  - Validates query content is geo/mineral domain before processing.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional

from services.rag_service import RAGService, RAGQuery
from models.schemas import QueryType
from utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

# Metrics DB path — persists in project root/data/
_METRICS_DB = Path(__file__).resolve().parent.parent.parent.parent / "data" / "query_metrics.db"


def _init_metrics_db():
    _METRICS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_METRICS_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS query_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            query TEXT NOT NULL,
            query_type TEXT NOT NULL,
            model TEXT,
            chunks_retrieved INTEGER,
            sources_count INTEGER,
            top_similarity REAL,
            answer_length INTEGER,
            latency_ms REAL,
            needs_clarification INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def _save_metric(record: dict):
    try:
        _init_metrics_db()
        conn = sqlite3.connect(str(_METRICS_DB))
        conn.execute("""
            INSERT INTO query_metrics
              (timestamp, query, query_type, model, chunks_retrieved, sources_count,
               top_similarity, answer_length, latency_ms, needs_clarification)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            record.get("timestamp"),
            record.get("query"),
            record.get("query_type"),
            record.get("model"),
            record.get("chunks_retrieved"),
            record.get("sources_count"),
            record.get("top_similarity"),
            record.get("answer_length"),
            record.get("latency_ms"),
            int(record.get("needs_clarification", False)),
        ))
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning(f"Metrics save failed: {exc}")


def _get_metrics(limit: int = 100) -> list[dict]:
    try:
        _init_metrics_db()
        conn = sqlite3.connect(str(_METRICS_DB))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM query_metrics ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning(f"Metrics fetch failed: {exc}")
        return []


def _delete_metric(metric_id: int) -> bool:
    try:
        conn = sqlite3.connect(str(_METRICS_DB))
        conn.execute("DELETE FROM query_metrics WHERE id=?", (metric_id,))
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


class QueryRequestBody(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    query_type: QueryType = Field(default=QueryType.ALL)
    top_k: Optional[int] = Field(default=10, ge=1, le=20)
    stream: bool = Field(default=False)
    source_filter: Optional[str] = Field(
        default=None,
        max_length=255,
        description=(
            "Restrict retrieval to a single ingested file (e.g. 'survey_2024.csv'). "
            "When set, chunks from any other file are excluded at the database "
            "level — their similarity scores will never appear in 'sources'."
        ),
    )


def get_rag_service(request: Request) -> RAGService:
    return request.app.state.container.rag_service


@router.post("/", summary="Ask a question over the ingested knowledge base")
async def query(body: QueryRequestBody, rag: RAGService = Depends(get_rag_service)):
    rag_query = RAGQuery(
        query=body.query,
        query_type=body.query_type.value,
        top_k=body.top_k,
        source_filter=body.source_filter,
    )

    if body.stream:
        def generate():
            for token in rag.stream_query(rag_query):
                yield token
        return StreamingResponse(generate(), media_type="text/plain")

    try:
        t0 = time.perf_counter()
        result = rag.query(rag_query)
        latency_ms = (time.perf_counter() - t0) * 1000

        top_sim = max((s["similarity"] for s in result.sources), default=0.0)

        # Persist metric
        _save_metric({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "query": result.query,
            "query_type": body.query_type.value,
            "model": result.model,
            "chunks_retrieved": result.chunks_retrieved,
            "sources_count": len(result.sources),
            "top_similarity": round(top_sim, 4),
            "answer_length": len(result.answer),
            "latency_ms": round(latency_ms, 1),
            "needs_clarification": result.needs_clarification,
        })

        return {
            "query": result.query,
            "answer": result.answer,
            "sources": result.sources,
            "chunks_retrieved": result.chunks_retrieved,
            "model": result.model,
            "clarifying_questions": result.clarifying_questions,
            "needs_clarification": result.needs_clarification,
        }
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Metrics dashboard endpoints ───────────────────────────────────────────────

@router.get("/metrics-history", summary="Persistent query metrics dashboard data")
async def get_metrics_history(limit: int = 100):
    """Return persistent query metrics stored in SQLite (survives restarts)."""
    rows = _get_metrics(limit)
    return {"metrics": rows, "total": len(rows)}


@router.delete("/metrics-history/{metric_id}", summary="Delete a metrics record")
async def delete_metric_record(metric_id: int):
    """Manually delete a specific metrics record."""
    ok = _delete_metric(metric_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Metric {metric_id} not found")
    return {"success": True, "deleted_id": metric_id}


@router.delete("/metrics-history", summary="Clear all metrics history")
async def clear_metrics_history():
    """Clear all persisted query metrics."""
    try:
        conn = sqlite3.connect(str(_METRICS_DB))
        conn.execute("DELETE FROM query_metrics")
        conn.commit()
        conn.close()
        return {"success": True, "message": "All metrics cleared"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/rock-formation", summary="Specialized query for rock formation interpretation")
async def query_rock_formation(description: str, rag: RAGService = Depends(get_rag_service)):
    rag_query = RAGQuery(
        query=f"Describe and interpret the following rock formation: {description}",
        query_type="geological",
        top_k=5,
    )
    result = rag.query(rag_query)
    return result.__dict__


@router.post("/mineral-zone", summary="Specialized query for mineral zone identification")
async def query_mineral_zones(query_text: str, rag: RAGService = Depends(get_rag_service)):
    rag_query = RAGQuery(
        query=f"Identify potential mineral zones related to: {query_text}",
        query_type="all",
        top_k=8,
    )
    result = rag.query(rag_query)
    return result.__dict__