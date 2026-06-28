"""
RAGService — Core RAG Business Logic with Prometheus instrumentation
─────────────────────────────────────────────────────────────────────────────
RAG PIPELINE (5 explicit steps):
  Step 1 — RETRIEVE  : Adaptive multi-collection vector search
  Step 2 — RE-RANK   : Cross-encoder re-scoring (keyword boost fallback)
  Step 3 — GUARD     : Halt if context empty — anti-hallucination gate
  Step 4 — GENERATE  : Prompt + LLM call (respects settings.MAX_TOKENS)
  Step 5 — ASSEMBLE  : Deduplicated citations + final answer


"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Iterator

from core.config import settings as _global_settings
from repositories.vector_repository import VectorRepository, SearchResult
from repositories.llm_provider import LLMProvider
from utils.logger import setup_logger

from metrics.prometheus_metrics import (
    rag_queries_total,
    rag_query_duration_seconds,
    rag_chunks_retrieved,
    evaluation_scores,
    errors_total,
)

logger = setup_logger(__name__)


# ── Prompt templates ──────────────────────────────────────────────────────────

GEOLOGICAL_SYSTEM_PROMPT = """\
You are a senior geological AI assistant helping mineral exploration teams \
interpret multi-source exploration data. You are given relevant geological \
information extracted from uploaded reports and datasets.

STRICT INSTRUCTIONS:
1. Answer ONLY from the information provided below. Do NOT use outside knowledge.
2. NEVER mention "chunks", "context chunks", "retrieved context", "present chunks", \
   "available context", "based on the provided context", or any similar phrasing \
   in your answer. Write as if you have studied these documents deeply.
3. READ AND USE ALL the information provided — important data (grades, intercepts, \
   coordinates) may appear in later sections.
4. Be technically precise — use correct geological terminology and units (ppm, g/t, %, m, km).
5. For every specific claim, cite the source file, section, and page in brackets, \
   e.g. [Source: report.pdf | Section: Drilling Results | Page: 14].
6. Structure your answer with clear Markdown headings (##) where the question \
   has multiple parts or the answer covers multiple topics.
7. When comparing zones, samples, or drill holes, present data in a Markdown table \
   if more than 3 items are compared.
8. If the information provided is not sufficient to answer, say ONLY: \
   "The uploaded documents do not contain information about [topic]. \
   Please upload the relevant dataset or report." \
   Do NOT say anything about "chunks" or "context".
9. NEVER invent numbers, grades, coordinates, or conclusions not present in the information.
10. At the end of your answer, add a ## Data Gaps section ONLY if there are \
    specific aspects of the question that could not be answered from the uploads.

Geological information from uploaded documents ({n_chunks} sections from {n_sources} file(s)):
{context}

Question: {query}

Answer:"""

DECISION_SUPPORT_PROMPT = """\
You are a mineral exploration strategist advising on prospect prioritisation \
and work-program design. You are given information from uploaded exploration datasets \
and reports.

STRICT INSTRUCTIONS:
1. Use ONLY the information provided below — cite every specific claim with \
   [Source | Section | Page].
2. READ ALL information before writing — critical grade data and risk factors \
   may appear in later sections.
3. Never invent grades, tonnages, coordinates, or recommendations not supported \
   by the uploaded documents.
4. NEVER mention "chunks", "context chunks", "retrieved context", "present chunks", \
   or similar technical terms in your answer. Write as an expert analyst.
5. Produce the full structured recommendation below — all 5 sections are required.

Information from uploaded documents ({n_chunks} sections from {n_sources} file(s)):
{context}

Question: {query}

Structured recommendation:

## 1. Key Findings
(Bullet each major finding with its source citation. Include all relevant grades, \
widths, coordinates, and deposit model indicators.)

## 2. Risk Matrix
| Risk Category | Description | Severity (H/M/L) | Mitigation |
|---|---|---|---|
(Fill one row per risk: grade continuity, structural, metallurgical, \
logistical, data-gap, environmental)

## 3. Recommended Work Program (phased)
**Phase 1 — Immediate (0–3 months):**
(Most critical, lowest-cost actions first)

**Phase 2 — Short-term (3–12 months):**
(Follow-up drilling, resource estimation steps)

**Phase 3 — Long-term (12+ months):**
(Feasibility, permitting, resource definition)

## 4. Exploration Priority Rating
**Rating: HIGH / MEDIUM / LOW**
Justification: (one paragraph citing supporting evidence)

## 5. Data Gaps
(List specific missing information that would change the recommendation if available)"""

CLARIFYING_QUESTIONS_PROMPT = """\
A user asked the following question to a mineral exploration AI system:

"{query}"

The system found some relevant information but the question is broad or ambiguous \
and could benefit from clarification to give a more precise answer.

Generate exactly 3 concise clarifying questions that would help narrow the scope. \
Questions should be specific to mineral exploration, geology, and geochemistry.

Respond ONLY with valid JSON array of exactly 3 strings, no markdown, no preamble:
["Question 1?", "Question 2?", "Question 3?"]"""

EVALUATION_PROMPT = """\
You are evaluating the quality of a RAG answer for a mineral exploration system.

Criteria to score (each 0.0–1.0):
  - relevance    : Does the answer directly address the question asked?
  - faithfulness : Is every claim in the answer supported by the provided information?
  - completeness : Are all aspects of the question addressed with appropriate detail?
  - conciseness  : Is the answer free of unnecessary padding, repetition, or off-topic content?

Scoring guide:
  1.0 = perfect, 0.8 = good minor issues, 0.5 = partially meets criterion, 0.2 = mostly fails, 0.0 = complete failure

Question: {query}

Retrieved information:
{context}

Answer being evaluated:
{answer}

Respond ONLY with valid JSON, no markdown, no extra text:
{{
  "relevance":    {{"score": 0.0, "explanation": "..."}},
  "faithfulness": {{"score": 0.0, "explanation": "..."}},
  "completeness": {{"score": 0.0, "explanation": "..."}},
  "conciseness":  {{"score": 0.0, "explanation": "..."}}
}}"""


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RAGQuery:
    query: str
    query_type: str = "all"   # "all" | "geological" | "mineral" | "decision"
    top_k: int = 10
    source_filter: str | None = None   # restrict retrieval to one dataset/report


@dataclass
class RAGAnswer:
    query: str
    answer: str
    sources: list[dict]
    chunks_retrieved: int
    model: str
    clarifying_questions: list[str] = field(default_factory=list)
    needs_clarification: bool = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_mentioned_files(query: str) -> list[str]:
    """Extract filenames mentioned in the query (with or without extension)."""
    # Match things like report.pdf, data.csv, myfile.json, or just quoted names
    pattern = r'\b[\w\-]+\.(pdf|csv|json|txt)\b'
    found = re.findall(pattern, query, re.IGNORECASE)
    # Also look for quoted names
    quoted = re.findall(r'"([^"]+)"', query) + re.findall(r"'([^']+)'", query)
    return [m[0] if isinstance(m, tuple) else m for m in re.findall(r'[\w\-]+\.\w+', query)] + quoted


def _is_ambiguous_query(query: str, chunks: list) -> bool:
    """Detect if a query is too broad/ambiguous and needs clarification."""
    if len(query.split()) < 5:
        return True
    ambiguous_patterns = [
        r'^(what|tell me|describe|explain|show|give).{0,20}(data|information|content|about)',
        r'^summarize',
        r'^what (do|does|is|are) (this|the|my)',
    ]
    q_lower = query.lower()
    for p in ambiguous_patterns:
        if re.search(p, q_lower):
            return True
    # If top chunk similarity is low across the board
    if chunks and max(c.similarity for c in chunks) < 0.25:
        return True
    return False


# ── Service ───────────────────────────────────────────────────────────────────

class RAGService:

    def __init__(
        self,
        vector_repo: VectorRepository,
        llm: LLMProvider,
        collection_map: dict[str, str],
        similarity_threshold: float = 0.0,
        max_context_chunks: int = 15,
        top_k: int = 10,
        reranker=None,
        hybrid_retriever=None,
    ):
        self._vectors              = vector_repo
        self._llm                  = llm
        self._collections          = collection_map
        self._similarity_threshold = similarity_threshold
        self._max_context_chunks   = max_context_chunks
        self._default_top_k        = top_k
        self._reranker             = reranker
        self._hybrid               = hybrid_retriever

    # ── Step 1: Retrieve ─────────────────────────────────────────────────────

    def _retrieve(self, query: RAGQuery) -> list[SearchResult]:
        def adaptive_top_k(collection: str, requested: int) -> int:
            total = self._vectors.count(collection)
            if total == 0:
                return requested
            return min(max(requested, max(int(total * 0.20), 5)), 60)

        if query.query_type in ("all", "decision"):
            unique_collections: list[str] = list(dict.fromkeys(self._collections.values()))
            all_results: list[SearchResult] = []
            for col in unique_collections:
                k = adaptive_top_k(col, query.top_k)
                try:
                    hits = self._vectors.search(
                        col, query.query, k, self._similarity_threshold,
                        source_filter=query.source_filter,
                    )
                    for r in hits:
                        r.collection = col
                    all_results.extend(hits)
                except Exception as exc:
                    logger.warning(f"Search failed for '{col}': {exc}")
            all_results.sort(key=lambda r: r.similarity, reverse=True)
            candidates = all_results[: query.top_k * 2]
            if self._hybrid:
                candidates = self._hybrid.fuse(
                    query=query.query,
                    dense_results=candidates,
                    top_k=query.top_k * 2,
                    source_filter=query.source_filter,
                )
            return candidates

        collection = self._collections.get(query.query_type)
        if not collection:
            logger.warning(f"Unknown query_type '{query.query_type}' — returning empty")
            return []

        k = adaptive_top_k(collection, query.top_k)
        results = self._vectors.search(
            collection, query.query, k, self._similarity_threshold,
            source_filter=query.source_filter,
        )
        for r in results:
            r.collection = collection

        # The "bonus" zone-comparison search below is a convenience widening
        # of recall for mineral queries; it must still respect source_filter,
        # otherwise a dataset-scoped query would leak chunks from other files
        # back in through this side door (this was part of Issue #3).
        if collection == self._collections.get("mineral"):
            bonus = self._vectors.search(
                collection,
                "zone comparison ranking summary statistics overview high grade priority",
                top_k=10,
                similarity_threshold=0.0,
                source_filter=query.source_filter,
            )
            seen_ids = {r.id for r in results}
            for r in bonus:
                if r.id not in seen_ids:
                    r.collection = collection
                    results.append(r)
                    seen_ids.add(r.id)

        if self._hybrid:
            results = self._hybrid.fuse(
                query=query.query,
                dense_results=results,
                top_k=query.top_k * 2,
                source_filter=query.source_filter,
            )
        return results
    # ── Step 2: Re-rank with file-name boosting ──────────────────────────────

    def _rerank(self, query: str, chunks: list[SearchResult]) -> list[SearchResult]:
        # Extract filenames mentioned in query for boosting
        mentioned_files = _extract_mentioned_files(query)
        mentioned_lower = [f.lower() for f in mentioned_files]

        if self._reranker is not None:
            ranked = self._reranker.rerank(query, chunks)
        else:
            terms = re.findall(
                r'\b([A-Z][a-z]{0,3}\d?|[A-Z]{2,4}|\d+\.?\d*\s*(?:ppm|g/t|%|m|km)|\w{5,})\b',
                query,
            )
            terms_lower = [t.lower() for t in terms]

            def boost(chunk: SearchResult) -> float:
                text_lower = chunk.text.lower()
                hits = sum(1 for t in terms_lower if t in text_lower)
                score = chunk.similarity + (hits * 0.025)
                # File-name boost: if a specific file is mentioned, strongly prefer those chunks
                if mentioned_lower:
                    src = chunk.metadata.get("source", "").lower()
                    if any(mf in src or src in mf for mf in mentioned_lower):
                        score += 0.3  # significant boost
                    else:
                        # Penalty for chunks from other files when a specific file is named
                        score -= 0.15
                return score

            ranked = sorted(chunks, key=boost, reverse=True)

        # If specific files were mentioned: filter out very low-scoring other-file chunks
        if mentioned_lower:
            primary = [c for c in ranked if any(
                mf in c.metadata.get("source", "").lower()
                for mf in mentioned_lower
            )]
            secondary = [c for c in ranked if c not in primary and c.similarity >= 0.35]
            ranked = (primary + secondary)[:self._max_context_chunks]

        return ranked

    # ── Step 4: Generate ─────────────────────────────────────────────────────

    def _build_prompt(self, query: RAGQuery, chunks: list[SearchResult]) -> str:
        top = chunks[: self._max_context_chunks]
        max_per_chunk = _global_settings.MAX_CONTEXT_CHARS_PER_CHUNK
        max_total = _global_settings.MAX_PROMPT_CONTEXT_CHARS

        context_parts = []
        total_len = 0
        used = 0
        for i, c in enumerate(top, 1):
            src  = c.metadata.get("source", "Unknown")
            sec  = c.metadata.get("section", c.metadata.get("doc_type", "Unknown"))
            page = c.metadata.get("page")
            page_str = f" | Page: {page}" if page else ""
            sim  = c.similarity

            # Issue #1 & #2 FIX: cap how much of any single chunk reaches the
            # prompt (chunks can be stored larger than this for retrieval
            # quality, but only a bounded slice is ever sent to the LLM), and
            # stop adding chunks once the total context budget is spent. This
            # keeps prompts from silently ballooning to 20k+ characters on
            # JSON/CSV datasets with many overview/zone/row chunks — the
            # single biggest cause of avoidable Groq token-rate-limit hits.
            text = c.text
            if len(text) > max_per_chunk:
                text = text[:max_per_chunk] + " …[truncated for length]"

            part = (
                f"[{i}] Source: {src} | Section: {sec}{page_str} | Relevance: {sim:.3f}\n"
                f"{text}"
            )
            if total_len + len(part) > max_total and used > 0:
                break
            context_parts.append(part)
            total_len += len(part)
            used += 1

        context = "\n\n---\n\n".join(context_parts)
        n_sources = len({c.metadata.get("source", "?") for c in top[:used]})

        if query.query_type == "decision":
            return DECISION_SUPPORT_PROMPT.format(
                n_chunks=used,
                n_sources=n_sources,
                context=context,
                query=query.query,
            )
        return GEOLOGICAL_SYSTEM_PROMPT.format(
            n_chunks=used,
            n_sources=n_sources,
            context=context,
            query=query.query,
        )

    # ── Step 5: Assemble ────────────────────────────────────────────────────

    @staticmethod
    def _format_sources(chunks: list[SearchResult]) -> list[dict]:
        sources, seen = [], set()
        for chunk in chunks:
            key = chunk.metadata.get("source", "Unknown")
            if key in seen:
                continue
            seen.add(key)
            snippet = chunk.text[:400] + "…" if len(chunk.text) > 400 else chunk.text
            sources.append({
                "source":     key,
                "doc_type":   chunk.metadata.get("doc_type", "Unknown"),
                "page":       chunk.metadata.get("page"),
                "section":    chunk.metadata.get("section"),
                "similarity": round(chunk.similarity, 4),
                "collection": chunk.collection,
                "snippet":    snippet,
            })
        return sources

    def _get_clarifying_questions(self, query: str) -> list[str]:
        """Generate clarifying questions when query is ambiguous."""
        import json
        try:
            prompt = CLARIFYING_QUESTIONS_PROMPT.format(query=query)
            raw = self._llm.generate(prompt, max_tokens=300).answer.strip()
            raw = re.sub(r'^```[a-z]*\n?', '', raw)
            raw = re.sub(r'\n?```$', '', raw)
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(q) for q in parsed[:3]]
        except Exception as exc:
            logger.warning(f"Clarifying questions failed: {exc}")
        return []

    # ── Public API ───────────────────────────────────────────────────────────

    def query(self, request: RAGQuery) -> RAGAnswer:
        """Execute the full 5-step RAG pipeline with Prometheus instrumentation."""
        qt = request.query_type
        rag_queries_total.labels(query_type=qt).inc()

        start = time.perf_counter()
        try:
            # Step 1
            chunks = self._retrieve(request)
            # Step 2
            chunks = self._rerank(request.query, chunks)
            # Step 3 — Guard
            if not chunks:
                logger.info(f"No context retrieved — LLM blocked. query={request.query!r}")
                rag_chunks_retrieved.labels(query_type=qt).observe(0)
                return RAGAnswer(
                    query=request.query,
                    answer=(
                        "No relevant information found in the knowledge base for this query. "
                        "Please upload related PDF reports or datasets first, then retry."
                    ),
                    sources=[],
                    chunks_retrieved=0,
                    model="none",
                )

            # Check if clarification needed
            needs_clarification = _is_ambiguous_query(request.query, chunks)
            clarifying_questions = []
            if needs_clarification:
                clarifying_questions = self._get_clarifying_questions(request.query)

            # Step 4
            prompt = self._build_prompt(request, chunks)
            result = self._llm.generate(prompt, max_tokens=_global_settings.MAX_TOKENS)
            # Step 5
            answer = RAGAnswer(
                query=request.query,
                answer=result.answer,
                sources=self._format_sources(chunks),
                chunks_retrieved=len(chunks),
                model=result.model,
                clarifying_questions=clarifying_questions,
                needs_clarification=needs_clarification and bool(clarifying_questions),
            )
        except Exception as exc:
            errors_total.labels(endpoint="/api/query", error_type=type(exc).__name__).inc()
            raise
        finally:
            elapsed = time.perf_counter() - start
            rag_query_duration_seconds.labels(query_type=qt).observe(elapsed)

        rag_chunks_retrieved.labels(query_type=qt).observe(len(chunks))
        return answer

    def stream_query(self, request: RAGQuery) -> Iterator[str]:
        """Stream response tokens."""
        chunks = self._retrieve(request)
        chunks = self._rerank(request.query, chunks)
        if not chunks:
            yield "No relevant information found in the knowledge base for this query."
            return
        prompt = self._build_prompt(request, chunks)
        yield from self._llm.stream(prompt, max_tokens=_global_settings.MAX_TOKENS)

    # ── Evaluation ───────────────────────────────────────────────────────────

    def evaluate(
        self,
        query: str,
        answer: str,
        context_chunks: list[str] | None = None,
    ) -> dict:
        """LLM-as-judge evaluation on four criteria."""
        import json

        context_text = "\n\n---\n\n".join(context_chunks or ["(no context provided)"])
        prompt = EVALUATION_PROMPT.format(
            query=query,
            context=context_text[:6000],
            answer=answer,
        )

        def _extract_json_object(text: str) -> dict | None:
            """
            Robustly pull a JSON object out of an LLM response that may
            contain markdown fences and/or leading/trailing prose, e.g.:
              "Here's my evaluation:\n```json\n{...}\n```\nLet me know..."
            Anchored regex (^```...$) only handles a fence with nothing
            else around it, which real judge models don't reliably produce
            despite "respond ONLY with JSON" instructions. This instead
            finds the first '{' and walks forward tracking brace depth
            (respecting strings) to find its true matching '}', then
            parses just that substring.
            """
            start = text.find("{")
            if start == -1:
                return None
            depth = 0
            in_string = False
            escape = False
            for i in range(start, len(text)):
                ch = text[i]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            return None
            return None

        _REQUIRED_CRITERIA = ("relevance", "faithfulness", "completeness", "conciseness")

        def _is_complete(obj: dict | None) -> bool:
            """
            A syntactically valid JSON object can still be a malformed
            evaluation if the judge truncated mid-generation or omitted a
            criterion — e.g. {"relevance": {"score": 1.0, ...}} with the
            other three keys missing. That silently defaults those scores
            to 0.0 (see below), which drags overall_score down for a
            reason that has nothing to do with answer quality. Treat that
            the same as a parse failure so it triggers the retry.
            """
            if not isinstance(obj, dict):
                return False
            for criterion in _REQUIRED_CRITERIA:
                data = obj.get(criterion)
                if not isinstance(data, dict) or "score" not in data:
                    return False
            return True

        def _call_judge() -> str:
            return self._llm.generate(prompt, max_tokens=1536).answer.strip()

        raw = _call_judge()
        parsed = _extract_json_object(raw)

        if not _is_complete(parsed):
            # One retry — transient malformed/truncated/incomplete output
            # from the judge model is common enough to be worth a single
            # extra attempt before giving up.
            logger.warning(f"Evaluation JSON incomplete or unparseable, retrying once: {raw[:200]}")
            raw = _call_judge()
            parsed = _extract_json_object(raw)

        if not _is_complete(parsed):
            logger.warning(f"Evaluation JSON incomplete or unparseable after retry: {raw[:200]}")
            return {
                "query": query,
                "answer": answer,
                "overall_score": 0.0,
                "error": "LLM returned malformed evaluation JSON.",
                "raw": raw,
            }

        THRESHOLD = 0.7
        results = []
        total = 0.0
        for criterion in _REQUIRED_CRITERIA:
            data = parsed.get(criterion, {})
            score = float(data.get("score", 0.0))
            total += score
            evaluation_scores.labels(criterion=criterion).observe(score)
            results.append({
                "criteria":    criterion,
                "score":       round(score, 3),
                "explanation": data.get("explanation", ""),
                "passed":      score >= THRESHOLD,
            })

        overall = round(total / len(results), 3)
        failing = [r["criteria"] for r in results if not r["passed"]]
        recommendation = (
            "Answer meets quality standards." if not failing
            else f"Improve these criteria: {', '.join(failing)}."
        )

        return {
            "query":          query,
            "answer":         answer,
            "overall_score":  overall,
            "results":        results,
            "recommendation": recommendation,
        }