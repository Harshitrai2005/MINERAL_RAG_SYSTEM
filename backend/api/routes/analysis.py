"""
Analysis Routes
─────────────────────────────────────────────────────────────────────────────
Mineral zone identification and exploration decision support.
Also hosts the RAG evaluation endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional

from services.rag_service import RAGService, RAGQuery
from utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()


# ── Request / response models ───────────────────────────────────────────────

class MineralZoneAnalysisRequest(BaseModel):
    data_summary: str = Field(..., min_length=10)
    include_report: bool = Field(default=False)


class MineralZoneResponse(BaseModel):
    analysis: str
    report: Optional[str] = None
    model: str


class EvaluationRequest(BaseModel):
    query: str = Field(..., min_length=3)
    answer: str = Field(..., min_length=5)
    context_chunks: list[str] = Field(default_factory=list)


# ── Dependency ──────────────────────────────────────────────────────────────

def get_rag_service(request: Request) -> RAGService:
    return request.app.state.container.rag_service


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/mineral-zones", response_model=MineralZoneResponse,
             summary="Identify potential mineral zones from provided data")
async def analyze_mineral_zones(
    body: MineralZoneAnalysisRequest,
    rag: RAGService = Depends(get_rag_service),
):
    try:
        rag_query = RAGQuery(query=body.data_summary, query_type="all", top_k=6)
        result = rag.query(rag_query)

        report = None
        if body.include_report:
            report_query = RAGQuery(
                query=f"Write a formal exploration report based on this analysis:\n{result.answer}",
                query_type="decision",
                top_k=4,
            )
            report = rag.query(report_query).answer

        return MineralZoneResponse(analysis=result.answer, report=report, model=result.model)

    except Exception as exc:
        logger.error(f"Mineral zone analysis failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/exploration-decision", summary="Get exploration decision support")
async def exploration_decision(
    context: str,
    target_commodity: str = "Au",
    rag: RAGService = Depends(get_rag_service),
):
    query_text = (
        f"Based on the following exploration context, provide a decision framework "
        f"for targeting {target_commodity} mineralization. "
        f"Include target prioritization, recommended work program, and key risk factors.\n\n"
        f"Context: {context}"
    )
    rag_query = RAGQuery(query=query_text, query_type="decision", top_k=8)
    try:
        result = rag.query(rag_query)
        return result.__dict__
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/deposit-models", summary="List supported deposit models")
async def list_deposit_models():
    """Return deposit model pathfinder suites understood by the system."""
    from ingestion.mineral_dataset_processor import DEPOSIT_PATHFINDERS
    return {
        "deposit_models": [
            {"name": model, "pathfinders": elements}
            for model, elements in DEPOSIT_PATHFINDERS.items()
        ],
    }


@router.post("/evaluate", summary="Evaluate RAG answer quality")
async def evaluate_answer(
    body: EvaluationRequest,
    rag: RAGService = Depends(get_rag_service),
):
    """
    Evaluate a RAG-generated answer on four criteria:
      - Relevance     : Does the answer address the question?
      - Faithfulness  : Is every claim grounded in the retrieved context?
      - Completeness  : Are all aspects of the question addressed?
      - Conciseness   : Is the answer appropriately brief (no fluff)?

    Each criterion is scored 0–1 by the LLM with an explanation.
    The endpoint returns an overall score and a pass/fail per criterion.
    """
    try:
        report = rag.evaluate(
            query=body.query,
            answer=body.answer,
            context_chunks=body.context_chunks,
        )
        return report
    except Exception as exc:
        logger.error(f"Evaluation failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
