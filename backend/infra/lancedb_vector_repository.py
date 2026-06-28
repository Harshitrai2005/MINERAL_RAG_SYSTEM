"""
LanceDB Vector Repository — Local File-Based Backend
─────────────────────────────────────────────────────────────────────────────
Default for local development and free-tier deployment (no external service
needed). Implements the exact same VectorRepository interface as Qdrant so
the rest of the app never notices which backend is active.

FIX (v5.1): Use cosine metric instead of L2 so similarity scores are correct.
  Old: similarity = 1 - L2_distance  →  WRONG (for unit vectors, L2 ∈ [0,2]
       so this maps to [-1, 1], breaking thresholds and re-ranking).
  New: metric="cosine" in LanceDB search  →  distance is already 1-cosine,
       so similarity = 1 - distance  is correct and ∈ [0, 1].
"""

from __future__ import annotations

import os
from typing import Optional

import pyarrow as pa
import lancedb

from repositories.vector_repository import VectorRepository, VectorDocument, SearchResult
from repositories.embedding_provider import EmbeddingProvider
from utils.logger import setup_logger

logger = setup_logger(__name__)


class LanceDBVectorRepository(VectorRepository):

    def __init__(self, persist_dir: str, embedder: EmbeddingProvider):
        os.makedirs(persist_dir, exist_ok=True)
        self._embedder = embedder
        self._db = lancedb.connect(persist_dir)
        self._tables: dict = {}

    # ── Schema ─────────────────────────────────────────────────────────────

    def _schema(self) -> pa.Schema:
        """
        Arrow schema for every collection.
        'source' is stored as a top-level column so we can filter it cheaply
        without deserializing the metadata blob.
        """
        return pa.schema([
            pa.field("id",          pa.string()),
            pa.field("text",        pa.string()),
            pa.field("source",      pa.string()),
            pa.field("doc_type",    pa.string()),
            pa.field("page",        pa.int32()),
            pa.field("section",     pa.string()),
            pa.field("file_hash",   pa.string()),
            pa.field("vector",      pa.list_(pa.float32(), self._embedder.dimension)),
        ])

    # ── Collection management ───────────────────────────────────────────────

    def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        if collection_name in self._tables:
            return
        if collection_name in self._db.table_names():
            self._tables[collection_name] = self._db.open_table(collection_name)
        else:
            self._tables[collection_name] = self._db.create_table(
                collection_name, schema=self._schema(), mode="create"
            )
            logger.debug(f"Created LanceDB table: {collection_name}")

    def _table(self, collection_name: str):
        self.ensure_collection(collection_name, self._embedder.dimension)
        return self._tables[collection_name]

    # ── Write ───────────────────────────────────────────────────────────────

    def add_documents(self, collection_name: str, documents: list[VectorDocument]) -> int:
        if not documents:
            return 0

        table = self._table(collection_name)
        texts = [d.text for d in documents]
        vectors = self._embedder.embed_texts(texts)

        rows = []
        for doc, vec in zip(documents, vectors):
            meta = doc.metadata
            rows.append({
                "id":        doc.id,
                "text":      doc.text,
                "source":    str(meta.get("source", "")),
                "doc_type":  str(meta.get("doc_type", "")),
                "page":      int(meta.get("page") or 0),
                "section":   str(meta.get("section", "")),
                "file_hash": str(meta.get("file_hash", "")),
                "vector":    [float(x) for x in vec],
            })
        table.add(rows)
        return len(rows)

    # ── Delete ──────────────────────────────────────────────────────────────

    def delete_by_source(self, collection_name: str, source_name: str) -> int:
        """
        Remove all rows whose 'source' column equals source_name.
        LanceDB delete() accepts a SQL-style predicate string.
        Returns the number of rows deleted (estimated via count before/after).
        """
        table = self._table(collection_name)
        before = table.count_rows()
        # Escape single quotes in the source name
        safe_name = source_name.replace("'", "''")
        table.delete(f"source = '{safe_name}'")
        after = table.count_rows()
        deleted = max(0, before - after)
        logger.info(f"Deleted {deleted} rows from '{collection_name}' for source '{source_name}'")
        return deleted

    # ── Read ────────────────────────────────────────────────────────────────

    def search(
        self,
        collection_name: str,
        query_text: str,
        top_k: int,
        similarity_threshold: float = 0.0,
        source_filter: Optional[str] = None,
    ) -> list[SearchResult]:
        table = self._table(collection_name)
        if table.count_rows() == 0:
            return []

        query_vector = self._embedder.embed_query(query_text)

        # FIX (v6.1 — Issue #3): when a specific dataset/source is requested,
        # apply a HARD pre-filter at the LanceDB query level (SQL WHERE on the
        # 'source' column) instead of post-hoc penalising other-file chunks
        # after retrieval. This means chunks from other reports/datasets are
        # never even candidates — they cannot appear in the response or in
        # the similarity-score list, no matter how high their score is.
        search_query = table.search(query_vector).metric("cosine")
        if source_filter:
            safe = source_filter.replace("'", "''").lower()
            # LanceDB/DataFusion SQL supports LOWER()+LIKE for case-insensitive
            # substring matching, so "report.pdf" also matches a filter of
            # "report" without requiring the exact stored filename.
            search_query = search_query.where(f"LOWER(source) LIKE '%{safe}%'", prefilter=True)

        # FIX: use cosine metric so _distance is (1 - cosine_similarity) ∈ [0, 1].
        # With L2 (old default): _distance ∈ [0, 2] and "1 - distance" is wrong
        # and goes negative for loosely related chunks, corrupting re-ranking and
        # thresholding. Cosine is correct because all-MiniLM-L6-v2 produces
        # unit-normalised embeddings where cosine similarity is the right metric.
        results = search_query.limit(top_k).to_pandas()

        matches = []
        for _, row in results.iterrows():
            # With cosine metric: _distance = 1 - cosine_similarity ∈ [0, 1]
            # so similarity = 1 - _distance = cosine_similarity ∈ [0, 1] — correct.
            distance = float(row.get("_distance", 1.0))
            similarity = max(0.0, 1.0 - distance)
            if similarity >= similarity_threshold:
                matches.append(SearchResult(
                    id=row["id"],
                    text=row["text"],
                    metadata={
                        "source":    row["source"],
                        "doc_type":  row["doc_type"],
                        "page":      int(row["page"]) if row["page"] != 0 else None,
                        "section":   row["section"],
                        "file_hash": row["file_hash"],
                    },
                    similarity=round(similarity, 4),
                    collection=collection_name,
                ))
        return matches

    def search_multi(
        self,
        collection_names: list[str],
        query_text: str,
        top_k: int,
        similarity_threshold: float = 0.0,
        source_filter: Optional[str] = None,
    ) -> list[SearchResult]:
        all_results: list[SearchResult] = []
        for name in collection_names:
            try:
                all_results.extend(
                    self.search(name, query_text, top_k, similarity_threshold, source_filter)
                )
            except Exception as exc:
                logger.warning(f"search_multi: collection '{name}' failed: {exc}")
        all_results.sort(key=lambda r: r.similarity, reverse=True)
        return all_results[:top_k]

    def count(self, collection_name: str) -> int:
        return self._table(collection_name).count_rows()

    def list_sources(self, collection_name: str) -> list[dict]:
        """Return distinct source files with chunk counts."""
        table = self._table(collection_name)
        if table.count_rows() == 0:
            return []
        df = table.to_pandas()[["source", "doc_type"]]
        grouped = df.groupby(["source", "doc_type"]).size().reset_index(name="chunk_count")
        return grouped.to_dict(orient="records")

    def health_check(self) -> bool:
        try:
            self._db.table_names()
            return True
        except Exception:
            return False