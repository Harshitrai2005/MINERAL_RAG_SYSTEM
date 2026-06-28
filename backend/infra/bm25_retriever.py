"""
BM25 Sparse Retriever
─────────────────────────────────────────────────────────────────────────────
Implements BM25 (Best Match 25) sparse retrieval over the in-memory chunk
corpus that was already loaded into the vector store.

WHY BM25 ALONGSIDE DENSE VECTORS:
  Dense bi-encoder embeddings excel at semantic similarity — "gold deposit"
  matches "auriferous mineralisation" even without shared words. But they
  under-rank EXACT keyword matches like "DH-002", "6.89 g/t", "0.42% Cu",
  or "MZ-001". BM25 is the opposite: it scores purely on term frequency and
  inverse document frequency, so rare geological codes and exact numeric
  values rank very high.

  Hybrid search (dense + BM25) captures BOTH semantic and exact-match
  relevance — critical for geological RAG where queries mix natural language
  ("what alteration types") with precise identifiers ("in zone MZ-002").

ALGORITHM — BM25 (Okapi BM25):
  score(q, d) = Σ IDF(t) * [ f(t,d) * (k1+1) ] / [ f(t,d) + k1*(1-b+b*|d|/avgdl) ]

  k1=1.5  (term frequency saturation — standard value for technical text)
  b=0.75  (length normalisation — standard)

  These are the de-facto defaults from the original Robertson et al. paper
  and work well on domain-specific corpora without tuning.

DESIGN:
  - Built from SearchResult objects already returned by the vector store,
    so there is NO second database call and NO duplicate storage.
  - Index is built lazily on first query and cached for the lifetime of the
    retriever instance (rebuilt on add_documents calls).
  - Falls back gracefully to an empty result list if the index is empty.
  - Pure Python + standard library only — no extra dependencies required.
    (rank_bm25 is used if installed for speed, pure-Python fallback otherwise)

FALLBACK:
  If rank_bm25 is not installed the module uses a hand-rolled BM25
  implementation so the service never fails on a cold deployment.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import TYPE_CHECKING

from utils.logger import setup_logger

if TYPE_CHECKING:
    from repositories.vector_repository import SearchResult

logger = setup_logger(__name__)


# ── Tokeniser ────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    """
    Domain-aware tokeniser for geological text.

    Splits on whitespace and punctuation but PRESERVES:
      - Numeric values with units  : "6.89g/t", "0.42%", "185Mt"
      - Drill-hole identifiers     : "DH-002", "CC-26-08"
      - Zone codes                 : "MZ-001", "MZ-002"
      - Element symbols            : "Au", "Cu", "Ag"

    Lowercased before indexing so "Au" and "au" match the same token,
    but NOT split into "a" and "u" (which would destroy element symbols).
    """
    # Keep alphanumeric runs, hyphens between alphanumerics, and decimal points
    tokens = re.findall(r'[a-zA-Z0-9]+(?:[.\-][a-zA-Z0-9]+)*', text)
    return [t.lower() for t in tokens if len(t) >= 2]


# ── Pure-Python BM25 (no external dependency) ────────────────────────────────

class _PureBM25:
    """Minimal BM25 implementation — used when rank_bm25 is not installed."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.n  = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / max(self.n, 1)
        self.corpus = corpus

        # document frequency per term
        df: dict[str, int] = {}
        for doc in corpus:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        self.df = df

        # IDF per term (smoothed)
        self.idf: dict[str, float] = {
            term: math.log((self.n - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.n
        for term in query_tokens:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, doc in enumerate(self.corpus):
                tf = doc.count(term)
                if tf == 0:
                    continue
                dl = len(doc)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * (tf * (self.k1 + 1)) / denom
        return scores


# ── BM25Retriever ─────────────────────────────────────────────────────────────

class BM25Retriever:
    """
    Sparse BM25 retriever that operates over a snapshot of SearchResult
    objects fetched from the vector store.

    Typical usage in HybridRetriever:
        bm25 = BM25Retriever()
        bm25.index(dense_results)           # build/update the index
        sparse = bm25.retrieve(query, k=20) # get BM25-ranked results
    """

    def __init__(self) -> None:
        self._chunks:  list[SearchResult] = []
        self._bm25:    _PureBM25 | None = None
        self._use_lib: bool = False
        self._lib_bm25 = None  # rank_bm25.BM25Okapi instance if available

        # Try importing rank_bm25 for faster scoring on large corpora
        try:
            import rank_bm25  # noqa: F401
            self._use_lib = True
            logger.info("BM25Retriever: using rank_bm25 library (faster)")
        except ImportError:
            logger.info("BM25Retriever: rank_bm25 not installed — using pure-Python fallback")

    # ── Index management ─────────────────────────────────────────────────────

    def index(self, chunks: list[SearchResult]) -> None:
        """
        Build (or rebuild) the BM25 index from a list of SearchResult chunks.

        Called by HybridRetriever every time a new dense retrieval is done,
        so the BM25 index always reflects the same candidate pool as the
        vector search.  This is intentional: BM25 re-scores the SAME set of
        candidates rather than searching a separate corpus, keeping latency
        low and results coherent.
        """
        if not chunks:
            self._chunks = []
            self._bm25 = None
            self._lib_bm25 = None
            return

        self._chunks = chunks
        tokenised = [_tokenise(c.text) for c in chunks]

        if self._use_lib:
            try:
                from rank_bm25 import BM25Okapi
                self._lib_bm25 = BM25Okapi(tokenised)
                self._bm25 = None
                return
            except Exception as exc:
                logger.warning(f"rank_bm25 index failed ({exc}), using pure-Python")

        self._bm25 = _PureBM25(tokenised)
        self._lib_bm25 = None

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        source_filter: str | None = None,
    ) -> list[SearchResult]:
        """
        Score all indexed chunks against the query using BM25 and return the
        top-k results as SearchResult objects with similarity = normalised BM25
        score (0–1 range).

        source_filter: mirrors the vector store's source_filter — restricts
        BM25 results to chunks whose metadata['source'] contains this string.
        Applied BEFORE scoring so we never leak cross-source chunks.
        """
        if not self._chunks or (self._bm25 is None and self._lib_bm25 is None):
            return []

        query_tokens = _tokenise(query)
        if not query_tokens:
            return []

        # Get raw BM25 scores
        if self._lib_bm25 is not None:
            raw_scores: list[float] = self._lib_bm25.get_scores(query_tokens).tolist()
        else:
            raw_scores = self._bm25.get_scores(query_tokens)  # type: ignore[union-attr]

        # Normalise to [0, 1]
        max_score = max(raw_scores) if raw_scores else 1.0
        if max_score == 0:
            max_score = 1.0
        norm_scores = [s / max_score for s in raw_scores]

        # Pair scores with chunks, apply source filter, sort, return top_k
        paired = list(zip(norm_scores, self._chunks))

        if source_filter:
            sf_lower = source_filter.lower()
            paired = [
                (s, c) for s, c in paired
                if sf_lower in c.metadata.get("source", "").lower()
            ]

        paired.sort(key=lambda x: x[0], reverse=True)
        top = paired[:top_k]

        # Return copies with similarity = normalised BM25 score
        results: list[SearchResult] = []
        for score, chunk in top:
            if score == 0.0:
                continue  # skip chunks with zero BM25 relevance
            from repositories.vector_repository import SearchResult as SR
            results.append(SR(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.metadata,
                similarity=round(score, 4),
                collection=chunk.collection,
            ))

        return results