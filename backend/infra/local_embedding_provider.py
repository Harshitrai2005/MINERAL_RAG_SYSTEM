"""
Local Sentence-Transformer Embedding Provider
─────────────────────────────────────────────────────────────────────────────
Concrete implementation of EmbeddingProvider using `sentence-transformers`,
running locally (no external API call, no API key needed, no per-token cost).

Model: all-MiniLM-L6-v2
  - 384-dimensional output
  - ~90MB on disk
  - Trained on 1B+ sentence pairs, strong general-purpose semantic similarity
  - Runs fast on CPU (no GPU required) — important for free-tier hosting
"""

from sentence_transformers import SentenceTransformer

from repositories.embedding_provider import EmbeddingProvider
from utils.logger import setup_logger

logger = setup_logger(__name__)


class LocalEmbeddingProvider(EmbeddingProvider):

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model '{model_name}'...")
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded — dimension={self._dimension}")

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, show_progress_bar=False, batch_size=32)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
