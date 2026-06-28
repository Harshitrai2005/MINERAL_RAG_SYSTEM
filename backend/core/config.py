"""
Core Configuration
─────────────────────────────────────────────────────────────────────────────
All settings sourced from environment variables with sensible defaults.

FIX (v5.1): Migrated from Pydantic V1 `class Config:` to V2 `model_config`
  (ConfigDict) and replaced deprecated `self.__fields__` with `model_fields`.
  Also increased quality defaults:
    - SIMILARITY_THRESHOLD: 0.0 → 0.15  (filter out truly unrelated chunks)
    - TOP_K_RESULTS: 6 → 10             (wider candidate pool for re-ranking)
    - MAX_TOKENS: 4096                   (now actually used by RAGService)
    - CHUNK_SIZE: 1200 → 1500           (larger chunks = more coherent context)
    - CHUNK_OVERLAP: 250 → 300          (more overlap = fewer cross-boundary misses)
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # ── API keys ────────────────────────────────────────────────────────────
    GROQ_API_KEY: str | None = None          # https://console.groq.com — free
    OPENAI_API_KEY: str | None = None        # https://platform.openai.com — fallback LLM
    API_KEY: str | None = None               # Backend API key for frontend authentication

    # ── App meta ────────────────────────────────────────────────────────────
    APP_NAME: str = "Mineral Exploration Intelligence System"
    VERSION: str = "5.1.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── LLM ─────────────────────────────────────────────────────────────────
    # "groq"   → GroqLLMProvider (default, free tier)
    # "openai" → OpenAILLMProvider (GPT-4o fallback)
    LLM_PROVIDER: str = "groq"
    # FIX: was 4096 in config but RAGService was hardcoding 2048 — now wired through
    MAX_TOKENS: int = 4096

    # ── Embeddings ──────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # ── Vector backend ───────────────────────────────────────────────────────
    # "lancedb" → local file-based (default, zero setup)
    # "qdrant"  → Qdrant Cloud (set QDRANT_URL + QDRANT_API_KEY)
    VECTOR_BACKEND: str = "lancedb"

    LANCEDB_PERSIST_DIR: str = str(_PROJECT_ROOT / "data" / "lancedb")

    QDRANT_URL: str | None = None
    QDRANT_API_KEY: str | None = None

    # ── Collections (2 only — imagery removed) ───────────────────────────────
    COLLECTION_GEOLOGICAL: str = "geological_reports"
    COLLECTION_MINERAL: str = "mineral_datasets"

    # ── Chunking ─────────────────────────────────────────────────────────────
    # FIX: increased chunk_size 1200→1500 and overlap 250→300 for better
    # coverage of large geological reports (500-page PDFs). Larger chunks
    # mean each vector captures more context; more overlap prevents facts
    # from being split across chunk boundaries.
    CHUNK_SIZE: int = 2_000       # v6: raised from 1500 for richer 100MB PDF coverage
    CHUNK_OVERLAP: int = 400      # v6: raised from 300 for fewer cross-boundary gaps

    # ── RAG ──────────────────────────────────────────────────────────────────
    # FIX: raised TOP_K_RESULTS 6→10 for wider candidate pool before re-ranking.
    TOP_K_RESULTS: int = 10
    # FIX: raised SIMILARITY_THRESHOLD 0.0→0.15 so genuinely unrelated chunks
    # (cosine similarity < 0.15) are filtered out before reaching the LLM.
    # With the L2 bug fixed (cosine metric now used), 0.15 is a meaningful cutoff.
    SIMILARITY_THRESHOLD: float = 0.15

    # ── Prompt / context budget (v6.1 — Issues #1 & #2) ──────────────────────
    # Groq's free tier rate-limits on TOKENS PER MINUTE, not just requests per
    # minute. A query that retrieves many large chunks (e.g. a big JSON
    # dataset with overview + zone + row chunks) can build a 20-30k character
    # prompt that alone eats most of the per-minute token budget, causing the
    # NEXT query to immediately hit a 429 even though it looks "unrelated".
    # These two caps bound prompt size independently of CHUNK_SIZE (which
    # controls how chunks are *stored*, not how much of each is *sent to the
    # LLM* at answer time):
    #   MAX_CONTEXT_CHARS_PER_CHUNK — hard cap per chunk when building the
    #     prompt; a chunk can still be stored larger, but only this many
    #     characters of it are ever sent to the LLM at once.
    #   MAX_PROMPT_CONTEXT_CHARS — hard cap on the TOTAL context block size
    #     (sum of all chunks) injected into the prompt, regardless of how
    #     many chunks were retrieved/re-ranked.
    # ~4 characters ≈ 1 token for English text, so 9,000 chars ≈ 2,250 tokens
    # of context — leaves comfortable headroom under typical free-tier Groq
    # per-minute token budgets even with MAX_TOKENS output on top.
    MAX_CONTEXT_CHARS_PER_CHUNK: int = 1_600
    MAX_PROMPT_CONTEXT_CHARS: int = 9_000

    # ── Re-ranking ───────────────────────────────────────────────────────────
    # "cross_encoder" → ms-marco-MiniLM-L-6-v2 (best quality, ~150ms CPU)
    # "keyword"       → lightweight keyword-boost heuristic (fast, no model)
    RERANKER: str = "cross_encoder"
 
    # ── Hybrid search (BM25 + dense vector fusion) ───────────────────────────
    # true  → RRF fusion of BM25 sparse + dense vector results (recommended)
    # false → dense-only retrieval (original behaviour, safe fallback)
    # Set HYBRID_SEARCH=false in .env to disable without touching code.
    HYBRID_SEARCH: bool = True

    # ── Files ────────────────────────────────────────────────────────────────
    UPLOAD_DIR: str = str(_PROJECT_ROOT / "data" / "uploads")
    # v6.1 note (Issue #4 — "what's the max file size this can handle well?"):
    # The hard server-side cap stays at 100 MB (generous, demo-impressive,
    # and the streaming CSV/JSON ingestion path can technically chew through
    # it). But "accepted" and "answers everything well on a free-tier Groq
    # key" are different claims. Practical sweet spots for a *portfolio demo*:
    #   PDF reports   : up to ~25-30 MB (a few hundred pages) — text extraction
    #                   + chunking handles this comfortably in well under a
    #                   minute locally.
    #   CSV datasets  : up to ~50 MB / ~250k rows — the chunked CSV reader
    #                   streams in batches so memory stays flat.
    #   JSON datasets : up to the ijson-streaming path's practical ceiling,
    #                   roughly ~50-80 MB of well-structured records; the
    #                   bounded json.load() fallback caps at 25 MB
    #                   (_JSON_FALLBACK_MAX_BYTES in mineral_dataset_processor.py)
    #                   for files that don't match the expected structure.
    #   TXT reports   : effectively unbounded by content (chunker streams
    #                   text), bounded only by MAX_UPLOAD_SIZE_MB.
    # Beyond these, ingestion still *works*, but answer latency and Groq
    # token/rate-limit pressure both rise because more chunks get embedded
    # and retrieved per query. For an interview demo: a 15-20 MB multi-page
    # PDF plus a 5-10k row CSV/JSON dataset is the size that reads as
    # "production-scale" without stressing the free-tier LLM quota.
    MAX_UPLOAD_SIZE_MB: int = 100        # generous for large PDF reports

    # ── Logging ──────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = str(_PROJECT_ROOT / "logs" / "app.log")

    # ── CORS ─────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: list[str] = ["*"]

    # ── Rate Limiting ────────────────────────────────────────────────────────
    # Requests per minute per IP address
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
    RATE_LIMIT_INGEST_REQUESTS_PER_MINUTE: int = 30

    # FIX: migrated from deprecated Pydantic V1 `class Config:` to V2 `model_config`
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Lowercase alias support: settings.chunk_size == settings.CHUNK_SIZE
    # FIX: replaced deprecated `self.__fields__` with `self.model_fields`
    def __getattr__(self, name: str):
        upper = name.upper()
        if upper in self.model_fields:
            return getattr(self, upper)
        raise AttributeError(f"Settings has no attribute '{name}'")

    @property
    def collection_map(self) -> dict[str, str]:
        """Maps category/query_type keys → physical collection names."""
        return {
            "geological": self.COLLECTION_GEOLOGICAL,
            "report":     self.COLLECTION_GEOLOGICAL,
            "mineral":    self.COLLECTION_MINERAL,
            "dataset":    self.COLLECTION_MINERAL,
        }


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()