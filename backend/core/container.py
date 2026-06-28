"""
Composition Root / Dependency Injection Container
─────────────────────────────────────────────────────────────────────────────
The ONLY file that imports concrete infrastructure classes. Everything else
depends only on abstract interfaces.

FIX (v5.1): LocalEmbeddingProvider is now lazily imported inside _build_embedder()
  instead of at module level. The old eager import caused:
    - All 15 route tests to fail with OSError (HuggingFace model download attempted
      at import time, before fakes were injected into app.state).
    - Slow startup in environments where the model isn't cached.
  This matches the pattern already used for QdrantVectorRepository, GroqLLMProvider,
  and OpenAILLMProvider — all of which are already lazily imported inside their
  builder functions.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.config import Settings
from repositories.vector_repository import VectorRepository
from repositories.llm_provider import LLMProvider
from repositories.embedding_provider import EmbeddingProvider
from services.rag_service import RAGService
from services.ingestion_service import IngestionService
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class AppContainer:
    """All fully-constructed components the app needs post-startup."""
    settings: Settings
    embedder: EmbeddingProvider
    vector_repo: VectorRepository
    llm: LLMProvider
    rag_service: RAGService
    ingestion_service: IngestionService


def _build_embedder(settings: Settings) -> EmbeddingProvider:
    # FIX: lazy import — only executed at runtime (startup), not at module import
    # time. This lets test_routes.py import main.py and inject FakeContainer
    # without triggering a HuggingFace model download.
    from infra.local_embedding_provider import LocalEmbeddingProvider
    return LocalEmbeddingProvider(model_name=settings.EMBEDDING_MODEL)


def _build_vector_repo(settings: Settings, embedder: EmbeddingProvider) -> VectorRepository:
    if settings.VECTOR_BACKEND == "qdrant":
        if not settings.QDRANT_URL or not settings.QDRANT_API_KEY:
            raise RuntimeError(
                "VECTOR_BACKEND=qdrant requires QDRANT_URL and QDRANT_API_KEY."
            )
        from infra.qdrant_vector_repository import QdrantVectorRepository
        logger.info(f"Vector backend: Qdrant Cloud ({settings.QDRANT_URL})")
        return QdrantVectorRepository(
            url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, embedder=embedder
        )

    from infra.lancedb_vector_repository import LanceDBVectorRepository
    logger.info(f"Vector backend: LanceDB (local) at {settings.LANCEDB_PERSIST_DIR}")
    return LanceDBVectorRepository(
        persist_dir=settings.LANCEDB_PERSIST_DIR, embedder=embedder
    )


def _build_llm(settings: Settings) -> LLMProvider:
    """
    Build the LLM provider based on LLM_PROVIDER env var.
    - "groq"   → GroqLLMProvider (default, free tier, fastest)
    - "openai" → OpenAILLMProvider (GPT-4o, best quality fallback)
    """
    provider = (settings.LLM_PROVIDER or "groq").lower()

    if provider == "openai":
        from infra.openai_llm_provider import OpenAILLMProvider
        if not settings.OPENAI_API_KEY:
            logger.warning(
                "LLM_PROVIDER=openai but OPENAI_API_KEY not set. "
                "Get a key at https://platform.openai.com/api-keys"
            )
        logger.info("LLM backend: OpenAI (GPT-4o)")
        return OpenAILLMProvider(api_key=settings.OPENAI_API_KEY or "")

    # Default: Groq
    from infra.groq_llm_provider import GroqLLMProvider
    if not settings.GROQ_API_KEY:
        logger.warning(
            "GROQ_API_KEY not set — LLM calls will fail. "
            "Get a free key at https://console.groq.com"
        )
    logger.info("LLM backend: Groq (llama-3.3-70b)")
    return GroqLLMProvider(api_key=settings.GROQ_API_KEY or "")


def _build_hybrid_retriever(settings: Settings):
    """
    Build the HybridRetriever if HYBRID_SEARCH=true (default).
    Returns None to disable hybrid search and keep dense-only retrieval.
    """
    if not getattr(settings, "HYBRID_SEARCH", True):
        logger.info("Hybrid search: disabled (HYBRID_SEARCH=false)")
        return None
    try:
        from infra.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever()
        logger.info("Hybrid search: BM25 + dense RRF fusion enabled")
        return retriever
    except Exception as exc:
        logger.warning(f"HybridRetriever init failed ({exc}) — dense-only fallback")
        return None

def _build_reranker(settings: Settings):
    """
    Build the re-ranker based on RERANKER env var.
    - "cross_encoder" → CrossEncoderReranker (ms-marco-MiniLM, best quality)
    - "keyword"       → None (RAGService uses built-in keyword-boost fallback)
    Returns None to signal keyword-boost fallback in RAGService.
    """
    reranker_type = (settings.RERANKER or "cross_encoder").lower()

    if reranker_type == "cross_encoder":
        try:
            from infra.cross_encoder_reranker import CrossEncoderReranker
            reranker = CrossEncoderReranker()
            if reranker.is_available:
                logger.info("Re-ranker: cross-encoder/ms-marco-MiniLM-L-6-v2")
                return reranker
            else:
                logger.warning("Cross-encoder unavailable — falling back to keyword boost.")
                return None
        except Exception as exc:
            logger.warning(f"Cross-encoder init failed ({exc}) — keyword boost fallback.")
            return None

    logger.info("Re-ranker: keyword-boost (heuristic)")
    return None


def build_container(settings: Settings) -> AppContainer:
    """Assemble every component exactly once at application startup."""
    embedder = _build_embedder(settings)
    vector_repo = _build_vector_repo(settings, embedder)

    # Pre-create collections so first query never hits a missing-table error
    for collection_name in set(settings.collection_map.values()):
        vector_repo.ensure_collection(collection_name, embedder.dimension)

    llm = _build_llm(settings)
    reranker = _build_reranker(settings)
    hybrid_retriever = _build_hybrid_retriever(settings)
 
    rag_service = RAGService(
        vector_repo=vector_repo,
        llm=llm,
        collection_map=settings.collection_map,
        similarity_threshold=settings.SIMILARITY_THRESHOLD,
        top_k=settings.TOP_K_RESULTS,
        reranker=reranker,
        hybrid_retriever=hybrid_retriever,
    )
    ingestion_service = IngestionService(
        vector_repo=vector_repo,
        collection_map=settings.collection_map,
    )

    logger.info("Application container built successfully.")
    return AppContainer(
        settings=settings,
        embedder=embedder,
        vector_repo=vector_repo,
        llm=llm,
        rag_service=rag_service,
        ingestion_service=ingestion_service,
    )
