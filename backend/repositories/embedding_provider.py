"""
Embedding Provider Interface
─────────────────────────────────────────────────────────────────────────────
Why this exists as its OWN interface, separate from VectorRepository:

Embedding (text -> vector) and Storage (vector -> persisted, searchable) are
two genuinely different concerns that happen to live together in naive RAG
implementations. Separating them means:
  1. We can swap the embedding model (MiniLM -> a bigger model) without
     touching how/where vectors are stored.
  2. We can swap the vector store (Qdrant -> Pinecone) without touching how
     embeddings are produced.
  3. It's independently unit-testable: assert that embed_texts(["gold ore"])
     returns a 384-length vector, with no database involved at all.

This is the Single Responsibility Principle applied at the architecture level.
"""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output vector size, e.g. 384 for all-MiniLM-L6-v2."""
        raise NotImplementedError

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed a list of strings into vectors."""
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        raise NotImplementedError
