"""
IngestionService — File Upload Orchestration with Prometheus metrics
─────────────────────────────────────────────────────────────────────────────
Routes an uploaded file to the correct domain processor (PDF / CSV-JSON),
then stores the resulting chunks via the VectorRepository interface.

Deliberately contains NO reference to Qdrant, LanceDB, or any concrete
infrastructure — only the abstract VectorRepository interface.
"""

from __future__ import annotations

import time
from pathlib import Path

from repositories.vector_repository import VectorRepository, VectorDocument
from ingestion.pdf_processor import PDFProcessor
from ingestion.mineral_dataset_processor import MineralDatasetProcessor
from utils.logger import setup_logger
from metrics.prometheus_metrics import (
    ingest_documents_total,
    ingest_duration_seconds,
    errors_total,
)

logger = setup_logger(__name__)

# ── Extension allow-list ───────────────────────────────────────────────────────
ALLOWED_EXTENSIONS: dict[str, set[str]] = {
    "report":  {".pdf", ".txt"},
    "dataset": {".csv", ".json"},
}


class UnsupportedFileTypeError(Exception):
    pass


class NoContentExtractedError(Exception):
    pass


class IngestionService:

    def __init__(self, vector_repo: VectorRepository, collection_map: dict[str, str]):
        self._vectors = vector_repo
        self._collections = collection_map   # {"report": "geological_reports", ...}
        self._pdf = PDFProcessor()
        self._dataset = MineralDatasetProcessor()

    # ── Validation ─────────────────────────────────────────────────────────

    def validate_extension(self, filename: str, category: str) -> None:
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS.get(category, set()):
            allowed = ", ".join(sorted(ALLOWED_EXTENSIONS.get(category, [])))
            raise UnsupportedFileTypeError(
                f"File type '{suffix}' not supported for category '{category}'. "
                f"Allowed: {allowed}"
            )

    # ── Processing ─────────────────────────────────────────────────────────

    def _process(self, file_path: Path, category: str) -> list[dict]:
        suffix = file_path.suffix.lower()
        if category == "report":
            if suffix == ".pdf":
                return self._pdf.process_file(file_path)
            return self._process_text_file(file_path)
        if category == "dataset":
            return self._dataset.process_file(file_path)
        return []

    @staticmethod
    def _process_text_file(file_path: Path) -> list[dict]:
        from utils.text_chunker import TextChunker
        from core.config import settings
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        chunker = TextChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        chunks = chunker.split(text)
        return [
            {
                "id": f"{file_path.stem}_c{i}",
                "text": c,
                "metadata": {
                    "source": file_path.name,
                    "doc_type": "text_report",
                    "section": "Unknown",
                    "file_hash": file_path.stem,
                },
            }
            for i, c in enumerate(chunks)
        ]

    # ── Public API ─────────────────────────────────────────────────────────

    def ingest_file(self, file_path: Path, category: str) -> dict:
        self.validate_extension(file_path.name, category)

        suffix = file_path.suffix.lower().lstrip(".")
        start = time.perf_counter()

        try:
            raw_chunks = self._process(file_path, category)
            if not raw_chunks:
                raise NoContentExtractedError(
                    f"No extractable content found in '{file_path.name}'. "
                    "The file may be empty, corrupted, or an unsupported internal format."
                )

            documents = [
                VectorDocument(id=c["id"], text=c["text"], metadata=c.get("metadata", {}))
                for c in raw_chunks
            ]

            collection_name = self._collections[category]
            count = self._vectors.add_documents(collection_name, documents)

            elapsed = time.perf_counter() - start
            ingest_documents_total.labels(doc_type=suffix, status="success").inc()
            ingest_duration_seconds.labels(doc_type=suffix).observe(elapsed)

            logger.info(f"Ingested '{file_path.name}' → {collection_name}: {count} chunks in {elapsed:.2f}s")
            return {
                "success": True,
                "file_name": file_path.name,
                "collection": collection_name,
                "chunks_added": count,
                "message": f"Successfully processed and indexed {count} chunks.",
            }

        except (UnsupportedFileTypeError, NoContentExtractedError):
            ingest_documents_total.labels(doc_type=suffix, status="error").inc()
            errors_total.labels(endpoint="/api/ingest/upload", error_type="ContentError").inc()
            raise
        except Exception as exc:
            elapsed = time.perf_counter() - start
            ingest_documents_total.labels(doc_type=suffix, status="error").inc()
            ingest_duration_seconds.labels(doc_type=suffix).observe(elapsed)
            errors_total.labels(endpoint="/api/ingest/upload", error_type=type(exc).__name__).inc()
            logger.error(f"Ingestion failed for '{file_path.name}': {exc}")
            raise

    def delete_file(self, source_name: str, category: str) -> dict:
        collection_name = self._collections.get(category)
        if not collection_name:
            raise ValueError(f"Unknown category: {category}")

        deleted = self._vectors.delete_by_source(collection_name, source_name)
        logger.info(f"Removed '{source_name}' from '{collection_name}': {deleted} chunks deleted")
        return {
            "success": True,
            "source_name": source_name,
            "collection": collection_name,
            "chunks_deleted": deleted,
            "message": f"Removed {deleted} chunks for '{source_name}'.",
        }

    def delete_by_file_type(self, file_type: str) -> dict:
        """
        Delete all chunks for every file matching the given extension (pdf/csv/json/txt).
        Scans both collections and removes any source whose filename ends with .{file_type}.
        Returns total chunks deleted and list of files removed.
        """
        file_type = file_type.lower().lstrip(".")
        total_deleted = 0
        removed_files: list[str] = []

        for collection_name in self._collections.values():
            try:
                sources = self._vectors.list_sources(collection_name)
            except Exception as exc:
                logger.warning(f"delete_by_file_type: could not list '{collection_name}': {exc}")
                continue

            for s in sources:
                src: str = s.get("source", "")
                if src.rsplit(".", 1)[-1].lower() == file_type:
                    try:
                        deleted = self._vectors.delete_by_source(collection_name, src)
                        total_deleted += deleted
                        removed_files.append(src)
                        logger.info(f"Deleted '{src}' from '{collection_name}': {deleted} chunks")
                    except Exception as exc:
                        logger.error(f"Failed to delete '{src}' from '{collection_name}': {exc}")

        return {
            "success": True,
            "file_type": file_type,
            "files_removed": removed_files,
            "files_count": len(removed_files),
            "chunks_deleted": total_deleted,
            "message": f"Removed {len(removed_files)} .{file_type} file(s), {total_deleted} chunks deleted.",
        }

    def list_indexed_files(self) -> list[dict]:
        all_files: list[dict] = []
        for category, collection_name in self._collections.items():
            try:
                sources = self._vectors.list_sources(collection_name)
                for s in sources:
                    all_files.append({**s, "category": category, "collection": collection_name})
            except Exception as exc:
                logger.warning(f"list_indexed_files: collection '{collection_name}' failed: {exc}")
        return all_files