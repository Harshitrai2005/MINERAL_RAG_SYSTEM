"""
Async Ingestion Queue — Background Worker
─────────────────────────────────────────────────────────────────────────────
Moves large-file ingestion off the HTTP request path so upload endpoints
return immediately (202 Accepted) and processing happens in the background.

PROBLEM WITHOUT THIS:
  Large PDFs (50-100MB) can take 10-30 seconds to chunk and embed.
  On a synchronous endpoint, the client waits — Render's 30s HTTP timeout
  kills the request, the user sees a 504, and nothing is indexed.

SOLUTION:
  POST /api/ingest/upload → enqueues job → returns 202 immediately
  Background worker dequeues → processes → updates job status
  GET /api/ingest/job/{job_id} → client polls for completion

DESIGN:
  - asyncio.Queue for in-process queuing (single-instance, zero dependencies)
  - Each job is a dataclass with status tracking (queued → processing → done/failed)
  - Worker runs as a background asyncio task started at app lifespan
  - For multi-instance deployments: swap the queue for Redis + Celery (same interface)

USAGE IN main.py (already wired via lifespan):
  from services.ingestion_queue import IngestionQueue
  queue = IngestionQueue(ingestion_service=container.ingestion_service)
  asyncio.create_task(queue.worker())   # start background worker
  app.state.ingestion_queue = queue     # expose for route injection
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from utils.logger import setup_logger

logger = setup_logger(__name__)


class JobStatus(str, Enum):
    QUEUED     = "queued"
    PROCESSING = "processing"
    DONE       = "done"
    FAILED     = "failed"


@dataclass
class IngestionJob:
    job_id: str
    file_path: Path
    category: str
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "job_id":       self.job_id,
            "status":       self.status.value,
            "category":     self.category,
            "file":         self.file_path.name,
            "created_at":   self.created_at,
            "completed_at": self.completed_at,
            "result":       self.result,
            "error":        self.error,
        }


class IngestionQueue:
    """
    Async background worker that processes ingestion jobs off the HTTP path.

    Instantiate once at startup, call worker() as an asyncio task,
    then use enqueue() from route handlers.
    """

    def __init__(self, ingestion_service, max_history: int = 200):
        self._service   = ingestion_service
        self._queue: asyncio.Queue[IngestionJob] = asyncio.Queue()
        self._jobs: dict[str, IngestionJob] = {}   # job_id → IngestionJob
        self._max_history = max_history            # cap in-memory job history

    # ── Public API ────────────────────────────────────────────────────────────

    def enqueue(self, file_path: Path, category: str) -> IngestionJob:
        """
        Add a file to the processing queue.
        Returns immediately with a job_id the client can poll.
        """
        job = IngestionJob(
            job_id=str(uuid.uuid4()),
            file_path=file_path,
            category=category,
        )
        self._jobs[job.job_id] = job
        self._queue.put_nowait(job)
        self._evict_old_jobs()
        logger.info(f"Enqueued ingestion job {job.job_id} for '{file_path.name}' ({category})")
        return job

    def get_job(self, job_id: str) -> Optional[IngestionJob]:
        return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> list[dict]:
        """Return the most recent jobs sorted by creation time (newest first)."""
        jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs[:limit]]

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    # ── Background worker ─────────────────────────────────────────────────────

    async def worker(self) -> None:
        """
        Infinite loop — run as an asyncio background task.
        Processes one job at a time (sequential to avoid I/O contention).
        Uses asyncio.to_thread so CPU-bound embedding doesn't block the event loop.
        """
        logger.info("Ingestion queue worker started.")
        while True:
            job = await self._queue.get()
            job.status = JobStatus.PROCESSING
            logger.info(f"Processing job {job.job_id}: {job.file_path.name}")
            try:
                # Run synchronous ingestion in a thread pool so FastAPI stays responsive
                result = await asyncio.to_thread(
                    self._service.ingest_file, job.file_path, job.category
                )
                job.status       = JobStatus.DONE
                job.result       = result
                job.completed_at = time.time()
                elapsed = job.completed_at - job.created_at
                logger.info(
                    f"Job {job.job_id} done in {elapsed:.1f}s: "
                    f"{result.get('chunks_added', 0)} chunks added"
                )
            except Exception as exc:
                job.status       = JobStatus.FAILED
                job.error        = str(exc)
                job.completed_at = time.time()
                logger.error(f"Job {job.job_id} failed: {exc}", exc_info=True)
            finally:
                self._queue.task_done()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _evict_old_jobs(self) -> None:
        """Keep the job history bounded to avoid unbounded memory growth."""
        if len(self._jobs) > self._max_history:
            oldest = sorted(self._jobs.values(), key=lambda j: j.created_at)
            for job in oldest[: len(self._jobs) - self._max_history]:
                if job.status in (JobStatus.DONE, JobStatus.FAILED):
                    del self._jobs[job.job_id]
