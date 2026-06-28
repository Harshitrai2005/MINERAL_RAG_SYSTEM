"""
Cross-Encoder Re-Ranker
─────────────────────────────────────────────────────────────────────────────
Replaces the keyword-boost heuristic in RAGService._rerank() with a proper
bi-encoder → cross-encoder two-stage pipeline.

STAGE 1 (retrieval)  — bi-encoder (sentence-transformers) — already done by
                        the vector store. Fast ANN search, ~100ms.
STAGE 2 (re-ranking) — cross-encoder (ms-marco-MiniLM-L-6-v2) — runs here.
                        Scores every (query, chunk) pair jointly, so the model
                        can attend to exact matches and subtle relevance cues
                        that cosine similarity on independent embeddings misses.
                        ~50-150ms for ≤20 chunks on CPU.

WHY THIS MATTERS FOR MINING RAG:
  Geological queries are highly specific ("0.52% Cu over 184 m in CC-26-08").
  Cosine similarity on dense embeddings under-ranks exact numeric matches.
  The cross-encoder sees the full pair and scores exact numeric proximity
  correctly — critical for assay data, drill intercept comparisons, etc.

FALLBACK:
  If the cross-encoder model fails to load (e.g., no internet, Render cold
  start), it falls back transparently to the keyword-boost heuristic so the
  service keeps running.

MODEL: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22M params, runs on CPU in <200ms for ≤20 passages
  - MS MARCO trained: question-answering retrieval, ideal for factual queries
  - Apache 2.0 licence, no API key required
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from utils.logger import setup_logger

if TYPE_CHECKING:
    from repositories.vector_repository import SearchResult

logger = setup_logger(__name__)

_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Two-stage re-ranker: loads the cross-encoder once at startup,
    then scores (query, chunk) pairs on demand.

    Usage in RAGService:
        self._reranker = CrossEncoderReranker()
        chunks = self._reranker.rerank(query, chunks, top_n=8)
    """

    def __init__(self, model_name: str = _CROSS_ENCODER_MODEL):
        self._model = None
        self._model_name = model_name
        self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name)
            logger.info(f"[OK] Cross-encoder loaded: {self._model_name}")
        except Exception as exc:
            logger.warning(
                f"Cross-encoder failed to load ({exc}). "
                "Falling back to keyword-boost re-ranking."
            )
            self._model = None

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def rerank(
        self,
        query: str,
        chunks: list["SearchResult"],
        top_n: int | None = None,
    ) -> list["SearchResult"]:
        """
        Re-rank chunks by cross-encoder score.

        Args:
            query:  The user's natural-language query.
            chunks: Candidate chunks from vector retrieval (already filtered
                    by similarity threshold).
            top_n:  If set, return only the top-n chunks. None = return all.

        Returns:
            Chunks sorted by cross-encoder score (descending).
            Falls back to keyword-boost ordering if model is unavailable.
        """
        if not chunks:
            return chunks

        if self._model is None:
            return _keyword_boost_fallback(query, chunks, top_n)

        try:
            pairs = [(query, chunk.text) for chunk in chunks]
            scores: list[float] = self._model.predict(pairs).tolist()

            scored = sorted(
                zip(scores, chunks),
                key=lambda x: x[0],
                reverse=True,
            )

            reranked = [chunk for _, chunk in scored]
            logger.debug(
                f"Cross-encoder re-ranked {len(chunks)} chunks "
                f"(top score: {scored[0][0]:.3f})"
            )
            return reranked[:top_n] if top_n else reranked

        except Exception as exc:
            logger.warning(f"Cross-encoder scoring failed ({exc}), using keyword fallback.")
            return _keyword_boost_fallback(query, chunks, top_n)


def _keyword_boost_fallback(
    query: str,
    chunks: list["SearchResult"],
    top_n: int | None,
) -> list["SearchResult"]:
    """
    Lightweight deterministic fallback — identical to the original
    RAGService keyword-boost heuristic. Preserved here so behaviour
    degrades gracefully if the cross-encoder model isn't available.
    """
    terms = re.findall(
        r'\b([A-Z][a-z]{0,2}\d?|[A-Z]{2,3}|\d+\.?\d*\s*(?:ppm|g/t|%|m|km)|\w{4,})\b',
        query,
    )
    terms_lower = [t.lower() for t in terms]

    def boost(chunk: "SearchResult") -> float:
        text_lower = chunk.text.lower()
        hits = sum(1 for t in terms_lower if t in text_lower)
        return chunk.similarity + (hits * 0.02)

    ranked = sorted(chunks, key=boost, reverse=True)
    return ranked[:top_n] if top_n else ranked
