"""
Hybrid Retriever — Dense + Sparse Fusion (Reciprocal Rank Fusion)
─────────────────────────────────────────────────────────────────────────────
Combines the dense vector search results (already produced by the vector
store) with BM25 sparse retrieval over the same candidate pool using
Reciprocal Rank Fusion (RRF).

WHY RRF (not score interpolation):
  Dense similarity scores (cosine, 0–1) and BM25 scores (TF-IDF weighted,
  unbounded) live on completely different scales.  Naively adding them
  together gives BM25 disproportionate weight on long documents.

  RRF sidesteps the scale problem entirely by working on RANKS, not scores:

    RRF(d) = Σ  1 / (k + rank_i(d))

  where k=60 is the standard smoothing constant (Cormack et al. 2009).
  A document ranked #1 in both lists gets:  1/61 + 1/61 ≈ 0.033
  A document ranked #1 in one and #10 in the other: 1/61 + 1/70 ≈ 0.030
  A document ranked #50 in both:  1/110 + 1/110 ≈ 0.018

  This makes top results from EITHER retrieval surface to the top of the
  fused list — which is exactly what we want for geological RAG where some
  queries are semantic ("what alteration types indicate porphyry?") and
  others are keyword-exact ("grades in DH-002").

HOW IT FITS INTO THE EXISTING PIPELINE:
  RAGService._retrieve() currently calls VectorRepository.search() and
  returns dense SearchResult objects.

  With HybridRetriever:
    1. Dense search runs as normal (unchanged).
    2. BM25Retriever.index(dense_results) is called — it re-scores the
       SAME candidates, so no extra DB call.
    3. RRF fuses both ranked lists into a single merged list.
    4. The merged list is passed to _rerank() as before — the cross-encoder
       then does its final quality pass on the hybrid-fused candidates.

  The result: exact numeric matches ("6.89 g/t", "MZ-001") surface from
  BM25, semantic matches surface from dense, and the cross-encoder picks
  the best of both worlds.

DESIGN CONSTRAINTS (so nothing else needs to change):
  - Input and output are both list[SearchResult] — same type RAGService
    already uses everywhere.
  - HybridRetriever is optional: if HYBRID_SEARCH=false in .env the
    container skips it and RAGService.query() works exactly as before.
  - No new abstract interfaces needed — this is a pure utility class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from infra.bm25_retriever import BM25Retriever
from utils.logger import setup_logger

if TYPE_CHECKING:
    from repositories.vector_repository import SearchResult

logger = setup_logger(__name__)

# RRF smoothing constant — 60 is the value from the original paper and is
# universally used as the default.  Higher k = less aggressive rank fusion.
_RRF_K = 60


def _reciprocal_rank_fusion(
    dense_results: list[SearchResult],
    sparse_results: list[SearchResult],
    k: int = _RRF_K,
) -> list[SearchResult]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    Returns a deduplicated list of SearchResult objects sorted by fused RRF
    score (descending).  The `similarity` field of each returned result is
    set to its normalised RRF score (0–1) so downstream code (cross-encoder,
    prompt builder, source formatter) can treat it identically to a plain
    dense similarity score.
    """
    # Build rank maps: chunk.id → 1-based rank in each list
    dense_rank:  dict[str, int] = {r.id: i + 1 for i, r in enumerate(dense_results)}
    sparse_rank: dict[str, int] = {r.id: i + 1 for i, r in enumerate(sparse_results)}

    # Union of all chunk ids
    all_ids = dict.fromkeys(
        [r.id for r in dense_results] + [r.id for r in sparse_results]
    )

    # Build a lookup from id → SearchResult (dense preferred for metadata)
    chunk_lookup: dict[str, SearchResult] = {}
    for r in sparse_results:
        chunk_lookup[r.id] = r
    for r in dense_results:
        chunk_lookup[r.id] = r  # dense overwrites sparse — keeps original metadata

    # Compute RRF score per chunk
    rrf_scores: dict[str, float] = {}
    for chunk_id in all_ids:
        score = 0.0
        if chunk_id in dense_rank:
            score += 1.0 / (k + dense_rank[chunk_id])
        if chunk_id in sparse_rank:
            score += 1.0 / (k + sparse_rank[chunk_id])
        rrf_scores[chunk_id] = score

    # Normalise RRF scores to [0, 1]
    max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0
    if max_rrf == 0:
        max_rrf = 1.0

    # Build sorted result list
    from repositories.vector_repository import SearchResult as SR

    fused: list[SearchResult] = []
    for chunk_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        chunk = chunk_lookup[chunk_id]
        fused.append(SR(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata,
            similarity=round(rrf_score / max_rrf, 4),  # normalised
            collection=chunk.collection,
        ))

    return fused


class HybridRetriever:
    """
    Drop-in hybrid retrieval layer for RAGService.

    Usage:
        retriever = HybridRetriever()
        # dense_results comes from VectorRepository.search() as normal
        fused = retriever.fuse(query, dense_results, source_filter=...)
        # fused is list[SearchResult], same type, same fields

    The retriever is stateless between queries — BM25 is re-indexed on each
    call from the dense candidate pool, keeping memory use minimal.
    """

    def __init__(self) -> None:
        self._bm25 = BM25Retriever()
        logger.info("HybridRetriever: BM25 + dense RRF fusion enabled")

    def fuse(
        self,
        query: str,
        dense_results: list[SearchResult],
        top_k: int = 20,
        source_filter: str | None = None,
    ) -> list[SearchResult]:
        """
        Given dense_results from the vector store, build a BM25 index over
        the same candidates and return an RRF-fused, deduplicated ranked list.

        Args:
            query:         User's natural-language query (passed to BM25).
            dense_results: Candidates from VectorRepository.search().
            top_k:         How many fused results to return.
            source_filter: Forwarded to BM25 to match vector store filtering.

        Returns:
            list[SearchResult] sorted by RRF score, length ≤ top_k.
        """
        if not dense_results:
            return []

        # Build BM25 index from this query's dense candidate pool
        self._bm25.index(dense_results)

        # BM25 sparse retrieval over the same pool
        sparse_results = self._bm25.retrieve(
            query=query,
            top_k=top_k,
            source_filter=source_filter,
        )

        if not sparse_results:
            # BM25 found nothing (e.g. all query terms are stop-words) —
            # fall back to dense-only, no fusion needed
            logger.debug("HybridRetriever: BM25 returned no results, using dense-only")
            return dense_results[:top_k]

        # Fuse with RRF
        fused = _reciprocal_rank_fusion(dense_results, sparse_results)

        logger.debug(
            f"HybridRetriever: dense={len(dense_results)} sparse={len(sparse_results)} "
            f"fused={len(fused)} top_k={top_k}"
        )

        return fused[:top_k]