"""
Qdrant Vector Repository — Cloud/Production Backend
─────────────────────────────────────────────────────────────────────────────
Implements the same VectorRepository interface as LanceDB so the rest of the
app never notices which backend is active.

Use Qdrant Cloud (free 1 GB cluster, no credit card) for persistent storage
on hosted platforms (Render, Railway) where the local filesystem is ephemeral.

Switch: set VECTOR_BACKEND=qdrant + QDRANT_URL + QDRANT_API_KEY in .env.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from repositories.vector_repository import VectorRepository, VectorDocument, SearchResult
from repositories.embedding_provider import EmbeddingProvider
from utils.logger import setup_logger

logger = setup_logger(__name__)


class QdrantVectorRepository(VectorRepository):

    def __init__(self, url: str, api_key: str, embedder: EmbeddingProvider):
        self._embedder = embedder
        self._client = QdrantClient(url=url, api_key=api_key, timeout=30)
        self._ensured: set[str] = set()

    # ── Collection management ───────────────────────────────────────────────

    def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        if collection_name in self._ensured:
            return
        existing = {c.name for c in self._client.get_collections().collections}
        if collection_name not in existing:
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection '{collection_name}'")
        self._ensured.add(collection_name)

    # ── Write ───────────────────────────────────────────────────────────────

    def add_documents(self, collection_name: str, documents: list[VectorDocument]) -> int:
        if not documents:
            return 0
        self.ensure_collection(collection_name, self._embedder.dimension)
        texts = [d.text for d in documents]
        vectors = self._embedder.embed_texts(texts)
        points = [
            qmodels.PointStruct(
                id=self._stable_point_id(doc.id),
                vector=vector,
                payload={"doc_id": doc.id, "text": doc.text, **doc.metadata},
            )
            for doc, vector in zip(documents, vectors)
        ]
        self._client.upsert(collection_name=collection_name, points=points)
        return len(points)

    # ── Delete ──────────────────────────────────────────────────────────────

    def delete_by_source(self, collection_name: str, source_name: str) -> int:
        """Delete all points whose payload.source == source_name."""
        self.ensure_collection(collection_name, self._embedder.dimension)
        before = self.count(collection_name)
        self._client.delete(
            collection_name=collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="source",
                            match=qmodels.MatchValue(value=source_name),
                        )
                    ]
                )
            ),
        )
        after = self.count(collection_name)
        deleted = max(0, before - after)
        logger.info(f"Deleted {deleted} points from '{collection_name}' for source '{source_name}'")
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
        self.ensure_collection(collection_name, self._embedder.dimension)
        if self.count(collection_name) == 0:
            return []
        query_vector = self._embedder.embed_query(query_text)

        # FIX (v6.1 — Issue #3): hard pre-filter by source at the Qdrant level.
        # Qdrant keyword payload fields only support exact match, so we use
        # MatchValue for an exact filename match. If the caller passes a
        # partial name, fall back to over-fetching and filtering client-side
        # so behaviour stays consistent with the LanceDB backend.
        qfilter = None
        if source_filter:
            qfilter = qmodels.Filter(
                must=[qmodels.FieldCondition(key="source", match=qmodels.MatchValue(value=source_filter))]
            )

        fetch_limit = top_k if not source_filter else max(top_k * 5, 50)
        hits = self._client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=qfilter,
            limit=fetch_limit,
            score_threshold=similarity_threshold if similarity_threshold > 0 else None,
        )
        results = [self._hit_to_result(h, collection_name) for h in hits]

        if source_filter and not qfilter:
            results = [r for r in results if source_filter.lower() in r.metadata.get("source", "").lower()]
        elif source_filter:
            # Exact match already filtered server-side; also catch case where
            # caller passed a partial name that doesn't exact-match anything —
            # retry with substring filtering client-side as a fallback.
            if not results:
                hits = self._client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=max(top_k * 5, 50),
                    score_threshold=similarity_threshold if similarity_threshold > 0 else None,
                )
                results = [
                    self._hit_to_result(h, collection_name) for h in hits
                    if source_filter.lower() in (h.payload or {}).get("source", "").lower()
                ]

        return results[:top_k]

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
                logger.warning(f"search_multi: '{name}' failed: {exc}")
        all_results.sort(key=lambda r: r.similarity, reverse=True)
        return all_results[:top_k]

    def count(self, collection_name: str) -> int:
        try:
            info = self._client.get_collection(collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    def list_sources(self, collection_name: str) -> list[dict]:
        """Scroll all points and aggregate unique sources with chunk counts."""
        self.ensure_collection(collection_name, self._embedder.dimension)
        counts: dict[str, dict] = {}
        offset = None
        while True:
            records, offset = self._client.scroll(
                collection_name=collection_name,
                limit=1000,
                offset=offset,
                with_payload=["source", "doc_type"],
                with_vectors=False,
            )
            for rec in records:
                src = rec.payload.get("source", "")
                if src not in counts:
                    counts[src] = {"source": src, "doc_type": rec.payload.get("doc_type", ""), "chunk_count": 0}
                counts[src]["chunk_count"] += 1
            if offset is None:
                break
        return list(counts.values())

    def health_check(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception as exc:
            logger.error(f"Qdrant health check failed: {exc}")
            return False

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _stable_point_id(doc_id: str) -> int:
        digest = hashlib.md5(doc_id.encode()).hexdigest()
        return int(digest[:16], 16)

    @staticmethod
    def _hit_to_result(hit, collection_name: str) -> SearchResult:
        payload = hit.payload or {}
        return SearchResult(
            id=payload.get("doc_id", str(hit.id)),
            text=payload.get("text", ""),
            metadata={k: v for k, v in payload.items() if k not in ("doc_id", "text")},
            similarity=round(float(hit.score), 4),
            collection=collection_name,
        )