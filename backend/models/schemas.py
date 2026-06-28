"""
Pydantic Schemas — Request / Response Models
"""
from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class QueryType(str, Enum):
    ALL        = "all"
    GEOLOGICAL = "geological"
    MINERAL    = "mineral"
    DECISION   = "decision"


class EvaluationCriteria(str, Enum):
    """
    Criteria available for the /api/analysis/evaluate endpoint.
    Each maps to a specialised prompt and scoring rubric inside rag_service.
    """
    RELEVANCE   = "relevance"     # Does the answer address the question?
    FAITHFULNESS = "faithfulness" # Is every claim grounded in the retrieved context?
    COMPLETENESS = "completeness" # Does the answer cover all aspects of the question?
    CONCISENESS  = "conciseness"  # Is the answer appropriately brief?


class EvaluationResult(BaseModel):
    criteria: EvaluationCriteria
    score: float = Field(..., ge=0.0, le=1.0)
    explanation: str
    passed: bool                  # score >= threshold (0.7 default)


class EvaluationReport(BaseModel):
    query: str
    answer: str
    overall_score: float
    results: list[EvaluationResult]
    recommendation: str
