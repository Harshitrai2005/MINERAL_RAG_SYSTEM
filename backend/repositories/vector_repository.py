"""
Vector Repository Interface
─────────────────────────────────────────────────────────────────────────────
CONTRACT for any vector storage backend (Qdrant, LanceDB, Pinecone, pgvector).
High-level services only ever depend on this interface — never on a concrete
client library (Dependency Inversion Principle, SOLID).

Swapping backends = writing ONE new class that implements this interface.
Zero changes elsewhere.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VectorDocument:
    """A single chunk of text + its metadata, ready to be embedded and stored."""
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """A single retrieved chunk with its similarity score."""
    id: str
    text: str
    metadata: dict
    similarity: float
    collection: str = ""


class VectorRepository(ABC):
    """Abstract interface every vector store backend must implement."""

    @abstractmethod
    def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        """Create the collection/table if it doesn't already exist (idempotent)."""

    @abstractmethod
    def add_documents(self, collection_name: str, documents: list[VectorDocument]) -> int:
        """Embed and upsert documents. Returns count inserted."""

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_text: str,
        top_k: int,
        similarity_threshold: float = 0.0,
        source_filter: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Semantic search within a single collection.

        source_filter: if provided, restricts the search to rows whose
        'source' field matches (case-insensitive substring match). This is
        a HARD pre-filter applied at the database level — non-matching
        documents are never scored or returned, so their similarity scores
        can never leak into the result set for a different dataset.
        """

    @abstractmethod
    def search_multi(
        self,
        collection_names: list[str],
        query_text: str,
        top_k: int,
        similarity_threshold: float = 0.0,
        source_filter: Optional[str] = None,
    ) -> list[SearchResult]:
        """Semantic search across multiple collections, merged and re-ranked."""

    @abstractmethod
    def count(self, collection_name: str) -> int:
        """Number of documents currently stored in a collection."""

    @abstractmethod
    def delete_by_source(self, collection_name: str, source_name: str) -> int:
        """
        Delete all chunks whose metadata.source == source_name.
        Returns the number of records deleted.

        This is the backing operation for the "Remove document" UI button.
        Implementations MUST match on the 'source' column/field stored in
        metadata at ingestion time.
        """

    @abstractmethod
    def list_sources(self, collection_name: str) -> list[dict]:
        """
        Return distinct source files indexed in a collection.
        Each entry: { source, doc_type, chunk_count }
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Return True if the underlying store is reachable."""