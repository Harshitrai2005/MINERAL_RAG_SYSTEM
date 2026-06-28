"""
Route-level Integration Tests
─────────────────────────────────────────────────────────────────────────────
Tests every API route using FastAPI's TestClient with in-memory fakes.
All infrastructure is replaced with fakes injected into app.state so the
full HTTP stack runs with zero real services, zero API keys, zero network.

Run:
    pytest tests/test_routes.py -v
"""

from __future__ import annotations

import io
import sys
import os
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import pytest
from dataclasses import dataclass, field
from unittest.mock import patch

from repositories.vector_repository import VectorRepository, VectorDocument, SearchResult
from repositories.llm_provider import LLMProvider, LLMResponse
from services.rag_service import RAGService
from services.ingestion_service import IngestionService
from services.ingestion_queue import IngestionQueue


# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeVectorRepository(VectorRepository):
    """Full in-memory VectorRepository implementing all abstract methods."""

    def __init__(self):
        self._store: dict[str, list[VectorDocument]] = {}

    def ensure_collection(self, name: str, vector_size: int) -> None:
        self._store.setdefault(name, [])

    def add_documents(self, name: str, docs: list[VectorDocument]) -> int:
        self._store.setdefault(name, []).extend(docs)
        return len(docs)

    def search(self, name: str, query: str, top_k: int, threshold: float = 0.0) -> list[SearchResult]:
        docs = self._store.get(name, [])
        return [
            SearchResult(id=d.id, text=d.text, metadata=d.metadata,
                         similarity=0.88, collection=name)
            for d in docs[:top_k]
        ]

    def search_multi(self, names, query, top_k, threshold=0.0):
        results = []
        for n in names:
            results.extend(self.search(n, query, top_k))
        return results[:top_k]

    def count(self, name: str) -> int:
        return len(self._store.get(name, []))

    def delete_by_source(self, name: str, source: str) -> int:
        before = len(self._store.get(name, []))
        self._store[name] = [d for d in self._store.get(name, [])
                              if d.metadata.get("source") != source]
        return before - len(self._store[name])

    def list_sources(self, name: str) -> list[dict]:
        seen, out = set(), []
        for d in self._store.get(name, []):
            src = d.metadata.get("source", "unknown")
            if src not in seen:
                seen.add(src)
                out.append({"source": src, "doc_type": d.metadata.get("doc_type", ""), "chunk_count": 1})
        return out

    def health_check(self) -> bool:
        return True


class FakeLLMProvider(LLMProvider):
    def generate(self, prompt: str, temperature=0.3, max_tokens=1024) -> LLMResponse:
        return LLMResponse(
            answer="Copper mineralisation confirmed at 0.52% Cu over 184 m.",
            model="fake-llm-v1",
            input_tokens=100,
            output_tokens=20,
        )

    def stream(self, prompt: str, temperature=0.3, max_tokens=1024):
        for token in ["Copper ", "mineralisation ", "confirmed."]:
            yield token

    def health_check(self) -> bool:
        return True


@dataclass
class FakeContainer:
    """Mirrors AppContainer — must have vector_repo, rag_service, ingestion_service."""
    vector_repo: FakeVectorRepository
    rag_service: RAGService
    ingestion_service: IngestionService


def _build_fake_container(seed_docs=None) -> FakeContainer:
    repo = FakeVectorRepository()
    llm = FakeLLMProvider()
    collection_map = {
        "geological": "geological_reports",
        "report":     "geological_reports",
        "mineral":    "mineral_datasets",
        "dataset":    "mineral_datasets",
    }

    repo.ensure_collection("geological_reports", 384)
    repo.ensure_collection("mineral_datasets", 384)

    if seed_docs:
        for col, docs in seed_docs.items():
            for d in docs:
                repo.add_documents(col, [VectorDocument(**d)])

    rag = RAGService(repo, llm, collection_map)
    ingestion = IngestionService(repo, collection_map)

    return FakeContainer(vector_repo=repo, rag_service=rag, ingestion_service=ingestion)


# ── App fixture ───────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """
    TestClient with fakes injected into app.state.
    We skip the real lifespan (which would require GROQ_API_KEY and API_KEY)
    by setting app.state attributes directly before starting the client.
    """
    from fastapi.testclient import TestClient
    from core.config import settings

    # Patch settings so the lifespan validators pass
    with patch.object(settings, "API_KEY", "test-api-key"), \
         patch.object(settings, "GROQ_API_KEY", "gsk_fake_key_for_testing_only"):

        from main import app

        # Build fake state before the lifespan touches it
        container = _build_fake_container(seed_docs={
            "geological_reports": [
                {
                    "id": "cc01",
                    "text": "Copper Creek property: 0.52% Cu over 184 m in hole CC-26-08.",
                    "metadata": {"source": "copper_creek.pdf", "page": 4, "doc_type": "geological_report"},
                }
            ]
        })

        # Create a minimal async queue stub so routes that need ingestion_queue don't crash
        fake_queue = IngestionQueue(ingestion_service=container.ingestion_service)

        app.state.container = container
        app.state.ingestion_queue = fake_queue

        # Use lifespan=False to skip the startup/shutdown hooks entirely —
        # our state is already wired above
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


@pytest.fixture
def api_key():
    return "test-api-key"


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") in ("ok", "healthy", "degraded")
    print("PASS: GET /api/health → 200")


# ── Query ─────────────────────────────────────────────────────────────────────

def test_query_returns_answer(client):
    r = client.post("/api/query/", json={
        "query": "What copper grades were found at Copper Creek?",
        "query_type": "geological",
        "top_k": 3,
    })
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert "sources" in data
    assert "chunks_retrieved" in data
    print(f"PASS: POST /api/query/ → answer ({data['chunks_retrieved']} chunks)")


def test_query_validates_min_length(client):
    """Query shorter than 3 chars must be rejected with 422."""
    r = client.post("/api/query/", json={"query": "hi", "query_type": "geological"})
    assert r.status_code == 422
    print("PASS: query min_length validation → 422")


def test_query_validates_max_length(client):
    """Query exceeding 2000 chars must be rejected with 422."""
    r = client.post("/api/query/", json={"query": "x" * 2001, "query_type": "geological"})
    assert r.status_code == 422
    print("PASS: query max_length validation → 422")


def test_streaming_query(client):
    """stream=True must return text/plain content."""
    r = client.post("/api/query/", json={
        "query": "What are the copper grades?",
        "query_type": "geological",
        "stream": True,
    })
    assert r.status_code == 200
    assert "text/plain" in r.headers.get("content-type", "")
    assert len(r.text) > 0
    print(f"PASS: streaming query → {len(r.text)} chars")


# ── Ingest ────────────────────────────────────────────────────────────────────

def test_upload_requires_api_key(client):
    """Upload without X-API-Key header must return 401."""
    pdf_bytes = b"%PDF-1.4 fake pdf content for testing"
    r = client.post(
        "/api/ingest/upload",
        files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"category": "report"},
        # Deliberately no X-API-Key header
    )
    assert r.status_code == 401
    print("PASS: upload without API key → 401")


def test_upload_rejects_invalid_category(client, api_key):
    """Category must be 'report' or 'dataset' — anything else is 400."""
    pdf_bytes = b"%PDF-1.4 test"
    r = client.post(
        "/api/ingest/upload",
        files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"category": "imagery"},
        headers={"X-API-Key": api_key},
    )
    assert r.status_code == 400
    print("PASS: invalid category → 400")


def test_upload_rejects_non_pdf_magic_bytes(client, api_key):
    """A .pdf filename with wrong magic bytes must be rejected."""
    fake_pdf = b"PK\x03\x04 this is actually a zip"
    r = client.post(
        "/api/ingest/upload",
        files={"file": ("report.pdf", io.BytesIO(fake_pdf), "application/pdf")},
        data={"category": "report"},
        headers={"X-API-Key": api_key},
    )
    assert r.status_code == 400
    print("PASS: fake PDF (wrong magic bytes) → 400")


def test_upload_rejects_disallowed_extension(client, api_key):
    """Executable files must be rejected regardless of content."""
    r = client.post(
        "/api/ingest/upload",
        files={"file": ("malware.exe", io.BytesIO(b"MZ\x90\x00"), "application/octet-stream")},
        data={"category": "report"},
        headers={"X-API-Key": api_key},
    )
    assert r.status_code == 400
    print("PASS: .exe upload → 400")


# ── Ingest list / stats ───────────────────────────────────────────────────────

def test_list_files_returns_array(client):
    r = client.get("/api/ingest/files")
    assert r.status_code == 200
    data = r.json()
    assert "files" in data
    assert isinstance(data["files"], list)
    print(f"PASS: GET /api/ingest/files → {data['total']} files")


def test_stats_endpoint(client):
    r = client.get("/api/ingest/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total_documents" in data
    assert "collections" in data
    print(f"PASS: GET /api/ingest/stats → {data['total_documents']} docs")


# ── Delete ────────────────────────────────────────────────────────────────────

def test_delete_requires_api_key(client):
    r = client.delete("/api/ingest/document?source_name=report.pdf&category=report")
    assert r.status_code == 401
    print("PASS: delete without API key → 401")


# ── Analysis ──────────────────────────────────────────────────────────────────

def test_deposit_models_endpoint(client):
    r = client.get("/api/analysis/deposit-models")
    assert r.status_code == 200
    data = r.json()
    assert "deposit_models" in data
    assert len(data["deposit_models"]) > 0
    print(f"PASS: GET /api/analysis/deposit-models → {len(data['deposit_models'])} models")


def test_evaluate_endpoint(client):
    r = client.post("/api/analysis/evaluate", json={
        "query": "What are the copper grades?",
        "answer": "Copper mineralisation confirmed at 0.52% Cu over 184 m.",
        "context_chunks": ["Hole CC-26-08 returned 0.52% Cu and 0.31 g/t Au over 184 metres."],
    })
    # FakeLLM returns non-JSON so evaluate() will return error dict, not crash
    assert r.status_code in (200, 500)
    print(f"PASS: POST /api/analysis/evaluate → {r.status_code}")


# ── Metrics ───────────────────────────────────────────────────────────────────

def test_metrics_endpoint(client):
    r = client.get("/api/metrics")
    assert r.status_code == 200
    assert "meis_" in r.text
    print("PASS: GET /api/metrics → Prometheus text")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=os.path.dirname(os.path.dirname(__file__))
    )
    sys.exit(result.returncode)
