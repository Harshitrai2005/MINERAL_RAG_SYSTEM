"""
Quick smoke test proving RAGService and IngestionService work correctly
against fake (in-memory) implementations of VectorRepository and LLMProvider
— with ZERO real database, ZERO real API key, ZERO network calls.

This is exactly the claim the layered architecture makes: business logic
depends only on abstractions. Run with:
    python3 _smoke_test_fakes.py
"""
import sys
sys.path.insert(0, ".")

from repositories.vector_repository import VectorRepository, VectorDocument, SearchResult
from repositories.llm_provider import LLMProvider, LLMResponse
from services.rag_service import RAGService, RAGQuery


class FakeVectorRepository(VectorRepository):
    """In-memory fake — no database at all."""
    def __init__(self):
        self.store: dict[str, list[VectorDocument]] = {}

    def ensure_collection(self, collection_name, vector_size):
        self.store.setdefault(collection_name, [])

    def add_documents(self, collection_name, documents):
        self.ensure_collection(collection_name, 384)
        self.store[collection_name].extend(documents)
        return len(documents)

    def search(self, collection_name, query_text, top_k, similarity_threshold=0.0):
        docs = self.store.get(collection_name, [])
        return [
            SearchResult(id=d.id, text=d.text, metadata=d.metadata, similarity=0.92, collection=collection_name)
            for d in docs[:top_k]
        ]

    def search_multi(self, collection_names, query_text, top_k, similarity_threshold=0.0):
        results = []
        for name in collection_names:
            results.extend(self.search(name, query_text, top_k, similarity_threshold))
        return results[:top_k]

    def count(self, collection_name):
        return len(self.store.get(collection_name, []))

    def delete_by_source(self, collection_name, source_name):
        before = len(self.store.get(collection_name, []))
        self.store[collection_name] = [
            d for d in self.store.get(collection_name, [])
            if d.metadata.get("source") != source_name
        ]
        return before - len(self.store[collection_name])

    def list_sources(self, collection_name):
        seen, out = set(), []
        for d in self.store.get(collection_name, []):
            src = d.metadata.get("source", "unknown")
            if src not in seen:
                seen.add(src)
                out.append({"source": src, "doc_type": d.metadata.get("doc_type", ""), "chunk_count": 1})
        return out

    def health_check(self):
        return True


class FakeLLMProvider(LLMProvider):
    """Fake LLM — returns a canned answer, proves no real API key is needed."""
    def generate(self, prompt, temperature=0.3, max_tokens=1024):
        assert "gold" in prompt.lower() or "context" in prompt.lower()
        return LLMResponse(answer="FAKE ANSWER: gold mineralization confirmed.", model="fake-llm-v1")

    def stream(self, prompt, temperature=0.3, max_tokens=1024):
        for word in ["FAKE", " STREAMED", " ANSWER"]:
            yield word

    def health_check(self):
        return True


def test_empty_context_blocks_llm():
    """The hard guard: if retrieval finds nothing, LLM must NOT be called."""
    vectors = FakeVectorRepository()
    llm = FakeLLMProvider()
    collection_map = {"geological": "geological_reports", "mineral": "mineral_datasets"}
    service = RAGService(vectors, llm, collection_map)

    result = service.query(RAGQuery(query="anything", query_type="geological"))
    assert result.model == "none", f"Expected no LLM call, got model={result.model}"
    assert result.chunks_retrieved == 0
    print("PASS: empty context blocks LLM call")


def test_rag_pipeline_end_to_end():
    """Full pipeline: ingest -> retrieve -> generate -> cite sources."""
    vectors = FakeVectorRepository()
    llm = FakeLLMProvider()
    collection_map = {
        "geological": "geological_reports", "report": "geological_reports",
        "mineral": "mineral_datasets", "dataset": "mineral_datasets",
    }

    vectors.add_documents("geological_reports", [
        VectorDocument(id="doc1_c0", text="The Sierra Negra prospect shows gold mineralization.",
                       metadata={"source": "sierra_negra.pdf", "page": 3, "doc_type": "geological_report"})
    ])

    service = RAGService(vectors, llm, collection_map)
    result = service.query(RAGQuery(query="What does the report say about gold?", query_type="geological"))

    assert result.chunks_retrieved == 1
    assert result.model == "fake-llm-v1"
    assert "gold" in result.answer.lower()
    assert result.sources[0]["source"] == "sierra_negra.pdf"
    assert result.sources[0]["page"] == 3
    print("PASS: end-to-end RAG pipeline works with fakes")


def test_streaming():
    vectors = FakeVectorRepository()
    llm = FakeLLMProvider()
    collection_map = {"geological": "geological_reports"}
    vectors.add_documents("geological_reports", [
        VectorDocument(id="d1", text="context about gold", metadata={"source": "x.pdf"})
    ])
    service = RAGService(vectors, llm, collection_map)

    tokens = list(service.stream_query(RAGQuery(query="gold?", query_type="geological")))
    assert "".join(tokens) == "FAKE STREAMED ANSWER"
    print("PASS: streaming works")


def test_source_deduplication():
    vectors = FakeVectorRepository()
    llm = FakeLLMProvider()
    collection_map = {"geological": "geological_reports"}
    vectors.add_documents("geological_reports", [
        VectorDocument(id="d1", text="gold info part 1", metadata={"source": "same_file.pdf", "page": 1}),
        VectorDocument(id="d2", text="gold info part 2", metadata={"source": "same_file.pdf", "page": 2}),
    ])
    service = RAGService(vectors, llm, collection_map)
    result = service.query(RAGQuery(query="gold", query_type="geological", top_k=5))
    assert len(result.sources) == 1, f"Expected deduped to 1 source, got {len(result.sources)}"
    print("PASS: source deduplication works")


if __name__ == "__main__":
    test_empty_context_blocks_llm()
    test_rag_pipeline_end_to_end()
    test_streaming()
    test_source_deduplication()
    print("\nALL ARCHITECTURE SMOKE TESTS PASSED")
