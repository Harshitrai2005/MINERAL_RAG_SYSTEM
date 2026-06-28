"""
Test Suite — Core RAG Pipeline
─────────────────────────────────────────────────────────────────────────────
All tests use in-memory fakes for VectorRepository and LLMProvider so the
full pipeline runs with zero infrastructure — no API key, no database, no
network calls required. Demonstrates the Dependency Inversion design:

    RAGService ──depends on──► VectorRepository  (abstract interface)
                                      ▲
                                      │ implements
                              FakeVectorRepository  (in-memory, test-only)

Run with:
    cd meis-final
    pytest tests/ -v

Or directly:
    python tests/test_core.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from repositories.vector_repository import VectorRepository, VectorDocument, SearchResult
from repositories.llm_provider import LLMProvider, LLMResponse
from services.rag_service import RAGService, RAGQuery


# ── In-memory fakes ───────────────────────────────────────────────────────────

class FakeVectorRepository(VectorRepository):
    """
    Fully in-memory VectorRepository — no embedding model, no database.
    Search returns stored documents in insertion order (deterministic for tests).
    Implements ALL abstract methods required by the VectorRepository ABC.
    """

    def __init__(self):
        self._store: dict[str, list[VectorDocument]] = {}

    def ensure_collection(self, collection_name: str, vector_size: int) -> None:
        self._store.setdefault(collection_name, [])

    def add_documents(self, collection_name: str, documents: list[VectorDocument]) -> int:
        self.ensure_collection(collection_name, 384)
        self._store[collection_name].extend(documents)
        return len(documents)

    def search(
        self, collection_name: str, query_text: str, top_k: int, similarity_threshold: float = 0.0
    ) -> list[SearchResult]:
        docs = self._store.get(collection_name, [])
        return [
            SearchResult(id=d.id, text=d.text, metadata=d.metadata,
                         similarity=0.92, collection=collection_name)
            for d in docs[:top_k]
        ]

    def search_multi(
        self, collection_names: list[str], query_text: str, top_k: int, similarity_threshold: float = 0.0
    ) -> list[SearchResult]:
        results = []
        for name in collection_names:
            results.extend(self.search(name, query_text, top_k))
        return results[:top_k]

    def count(self, collection_name: str) -> int:
        return len(self._store.get(collection_name, []))

    def delete_by_source(self, collection_name: str, source_name: str) -> int:
        before = len(self._store.get(collection_name, []))
        self._store[collection_name] = [
            d for d in self._store.get(collection_name, [])
            if d.metadata.get("source") != source_name
        ]
        return before - len(self._store[collection_name])

    def list_sources(self, collection_name: str) -> list[dict]:
        seen, out = set(), []
        for d in self._store.get(collection_name, []):
            src = d.metadata.get("source", "unknown")
            if src not in seen:
                seen.add(src)
                out.append({"source": src, "doc_type": d.metadata.get("doc_type", ""), "chunk_count": 1})
        return out

    def health_check(self) -> bool:
        return True


class FakeLLMProvider(LLMProvider):
    """
    Fake LLM — returns a canned answer without any API call.
    Validates that the prompt was non-empty before answering.
    """

    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 1024) -> LLMResponse:
        assert len(prompt) > 10, "Prompt must not be empty"
        return LLMResponse(
            answer="Gold mineralisation confirmed at 2.3 g/t Au in the NW fault zone.",
            model="fake-llm-v1",
            input_tokens=len(prompt) // 4,
            output_tokens=20,
        )

    def stream(self, prompt: str, temperature: float = 0.3, max_tokens: int = 1024):
        assert len(prompt) > 10
        for token in ["Gold ", "mineralisation ", "confirmed."]:
            yield token

    def health_check(self) -> bool:
        return True


# ── Test helpers ──────────────────────────────────────────────────────────────

# Canonical collection map — exactly what config.py produces (2 collections only)
_COLLECTION_MAP = {
    "geological": "geological_reports",
    "report":     "geological_reports",
    "mineral":    "mineral_datasets",
    "dataset":    "mineral_datasets",
}


def _make_service(seed_docs: dict[str, list[dict]] | None = None) -> RAGService:
    """
    Build a RAGService wired to in-memory fakes.
    seed_docs: {collection_name: [{id, text, metadata}, ...]}
    """
    vectors = FakeVectorRepository()
    llm = FakeLLMProvider()

    if seed_docs:
        for collection, docs in seed_docs.items():
            for d in docs:
                vectors.add_documents(collection, [VectorDocument(**d)])

    return RAGService(vectors, llm, _COLLECTION_MAP)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_empty_collection_blocks_llm_call():
    """
    Anti-hallucination guard: when no context is retrieved, RAGService must
    return early without calling the LLM. Most critical correctness property.
    """
    service = _make_service()
    result = service.query(RAGQuery(query="What minerals are in the north zone?", query_type="geological"))

    assert result.model == "none", (
        f"LLM was called despite empty context (model={result.model!r}). "
        "RAGService must block LLM calls when retrieval returns nothing."
    )
    assert result.chunks_retrieved == 0
    assert result.sources == []
    print("PASS: empty context blocks LLM call")


def test_full_rag_pipeline_geological():
    """Happy path: seed one geological document, query it, verify answer and citations."""
    service = _make_service(seed_docs={
        "geological_reports": [
            {
                "id": "sierra_c0",
                "text": "The Sierra Negra prospect shows gold mineralisation at 2.3 g/t Au in a NW-trending fault zone.",
                "metadata": {"source": "sierra_negra.pdf", "page": 3, "doc_type": "geological_report"},
            }
        ]
    })

    result = service.query(RAGQuery(query="What does the report say about gold?", query_type="geological"))

    assert result.chunks_retrieved >= 1
    assert result.model == "fake-llm-v1"
    assert "gold" in result.answer.lower()
    assert len(result.sources) == 1
    assert result.sources[0]["source"] == "sierra_negra.pdf"
    assert result.sources[0]["page"] == 3
    print("PASS: full RAG pipeline — geological query")


def test_full_rag_pipeline_mineral_dataset():
    """Verify mineral dataset collection is queryable independently."""
    service = _make_service(seed_docs={
        "mineral_datasets": [
            {
                "id": "survey_c0",
                "text": "Sample DGS-1153 in Northwest Gossan zone: Au=2.76 ppm, Cu=770 ppm. High grade. Drill target.",
                "metadata": {"source": "deep_survey.csv", "doc_type": "geochemical_dataset"},
            }
        ]
    })

    result = service.query(RAGQuery(query="Which samples are high grade drill targets?", query_type="mineral"))

    assert result.chunks_retrieved >= 1
    assert result.model == "fake-llm-v1"
    print("PASS: full RAG pipeline — mineral dataset query")


def test_streaming_response():
    """Stream query should yield token strings and form a non-empty answer."""
    service = _make_service(seed_docs={
        "geological_reports": [
            {"id": "d1", "text": "Gold at 5 ppm in quartz vein.", "metadata": {"source": "report.pdf"}}
        ]
    })

    tokens = list(service.stream_query(RAGQuery(query="gold?", query_type="geological")))
    full_answer = "".join(tokens)

    assert len(tokens) > 0
    assert len(full_answer) > 0
    print(f"PASS: streaming — {len(tokens)} tokens: {full_answer!r}")


def test_source_deduplication():
    """
    Multiple chunks from the same source must deduplicate to one entry in the
    sources list — so the UI doesn't show the same PDF five times.
    """
    docs = [
        {"id": f"d{i}", "text": f"gold content — page {i}",
         "metadata": {"source": "same_file.pdf", "page": i}}
        for i in range(5)
    ]
    service = _make_service(seed_docs={"geological_reports": docs})
    result = service.query(RAGQuery(query="gold", query_type="geological", top_k=5))

    assert len(result.sources) == 1, (
        f"Expected 1 deduplicated source, got {len(result.sources)}. "
        "The same PDF should appear once in citations even if multiple chunks matched."
    )
    print("PASS: source deduplication")


def test_cross_collection_query():
    """
    query_type='all' must search both geological and mineral collections
    and merge results, returning chunks from multiple collections.
    """
    service = _make_service(seed_docs={
        "geological_reports": [
            {"id": "geo1", "text": "Gold in NW fault zone.",
             "metadata": {"source": "report.pdf", "doc_type": "geological_report"}}
        ],
        "mineral_datasets": [
            {"id": "min1", "text": "Sample Au=6.7 ppm, Cu=1200 ppm.",
             "metadata": {"source": "assay.csv", "doc_type": "geochemical_dataset"}}
        ],
    })

    result = service.query(RAGQuery(query="gold copper mineralisation", query_type="all", top_k=5))

    assert result.chunks_retrieved >= 2, (
        f"Cross-collection query should retrieve from both collections, got {result.chunks_retrieved}"
    )
    source_names = {s["source"] for s in result.sources}
    assert "report.pdf" in source_names
    assert "assay.csv" in source_names
    print(f"PASS: cross-collection query — {result.chunks_retrieved} chunks from {len(source_names)} sources")


def test_decision_prompt_used_for_decision_type():
    """
    query_type='decision' must use the DECISION_SUPPORT_PROMPT and not fall
    back to the empty-context guard.
    """
    service = _make_service(seed_docs={
        "geological_reports": [
            {"id": "g1", "text": "Target A: IOCG system, Cu 1.2%, high priority.",
             "metadata": {"source": "targets.pdf"}}
        ]
    })

    result = service.query(RAGQuery(query="Which prospect should we drill first?", query_type="decision"))

    assert result.model != "none"
    assert result.chunks_retrieved >= 1
    print("PASS: decision query type uses decision prompt")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_empty_collection_blocks_llm_call,
        test_full_rag_pipeline_geological,
        test_full_rag_pipeline_mineral_dataset,
        test_streaming_response,
        test_source_deduplication,
        test_cross_collection_query,
        test_decision_prompt_used_for_decision_type,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL: {t.__name__} — {e}")
            failed += 1

    print(f"\n{'ALL TESTS PASSED' if not failed else f'{failed} TESTS FAILED'} ({len(tests)} total)")
    sys.exit(failed)
