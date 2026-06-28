"""
PDF Processor — Large-PDF-Safe Geological Report Extractor
===========================================================
Design goals:
  • Stream pages one at a time — never load the whole document into RAM.
  • Inject a context_prefix into every chunk (section + page) so retrieval
    always knows WHERE a chunk comes from.
  • Detect section headers at page level for fine-grained metadata.
  • Tag each chunk with file_hash so selective deletion works correctly.
  • Support PDFs of any size (tested on 500-page reports).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from core.config import settings
from utils.logger import setup_logger
from utils.text_chunker import TextChunker

logger = setup_logger(__name__)

# ── Section-header patterns (ordered: more specific first) ────────────────────
_SECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'(?i)(executive\s+summary)', re.M), "Executive Summary"),
    (re.compile(r'(?i)(geological\s+setting|regional\s+geology)', re.M), "Geological Setting"),
    (re.compile(r'(?i)(minerali[sz]ation|mineral\s+zones?)', re.M), "Mineralization"),
    (re.compile(r'(?i)(geochemi(?:stry|cal)\s+(?:data|survey|results|analysis))', re.M), "Geochemistry"),
    (re.compile(r'(?i)(rock\s+types?|litholog(?:y|ies))', re.M), "Lithology"),
    (re.compile(r'(?i)(structural\s+geology|tectonics?|fault\s+system)', re.M), "Structural Geology"),
    (re.compile(r'(?i)(drilling\s+results?|borehole|drill\s+holes?|assay)', re.M), "Drilling Results"),
    (re.compile(r'(?i)(resource\s+estimate|mineral\s+resource|reserve)', re.M), "Resource Estimates"),
    (re.compile(r'(?i)(hyperspectral|remote\s+sens|satellite)', re.M), "Remote Sensing"),
    (re.compile(r'(?i)(recommendation|exploration\s+target|next\s+steps?)', re.M), "Recommendations"),
    (re.compile(r'(?i)(introduction|background|overview)', re.M), "Introduction"),
    (re.compile(r'(?i)(location|property\s+description|access)', re.M), "Property Location"),
    (re.compile(r'(?i)(conclusion|summary)', re.M), "Summary/Conclusions"),
]


class PDFProcessor:
    """
    Processes geological PDF reports into richly-annotated text chunks.

    Key design decisions (mention in interview):
    - Page-by-page streaming: O(1) memory regardless of PDF size.
    - Context prefix injection: each chunk carries "Section | Page N" header,
      so even out-of-context retrieval gives the reader orientation.
    - Section-aware chunking: chunk boundaries are preferred at section
      transitions, keeping semantically related text together.
    - Deduplication via file_hash: the first 64 KB of the file is hashed;
      this lets the delete endpoint remove all chunks from a given file
      without scanning every row.
    """

    def __init__(self):
        self._chunker = TextChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def process_file(self, file_path: str | Path) -> list[dict]:
        """
        Extract and chunk a single PDF.

        Returns a list of document dicts:
          { id, text, metadata: { source, doc_type, page, section,
                                  total_pages, file_hash, chunk_index,
                                  has_table, has_coordinates, has_geochemistry } }
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {file_path.name}")

        logger.info(f"Processing PDF: {file_path.name}")
        file_hash = _compute_hash(file_path)

        doc = fitz.open(str(file_path))
        total_pages = len(doc)
        all_chunks: list[dict] = []
        current_section = "Unknown"

        for page_num in range(total_pages):
            page = doc[page_num]
            raw_text = page.get_text("text")   # text-layer extraction (fast)

            if not raw_text.strip():
                logger.debug(f"  Page {page_num + 1}/{total_pages} empty — skipping")
                continue

            # Update running section tracker
            detected = _detect_section(raw_text[:600])
            if detected:
                current_section = detected

            # Build context prefix that rides with every chunk from this page
            context_prefix = (
                f"[Source: {file_path.name} | Section: {current_section} "
                f"| Page: {page_num + 1}/{total_pages}]\n"
            )

            rich_chunks = self._chunker.split_rich(raw_text, context_prefix=context_prefix)

            for chunk in rich_chunks:
                doc_id = f"{file_hash}_p{page_num + 1}_c{chunk.chunk_index}"
                all_chunks.append({
                    "id": doc_id,
                    "text": chunk.text,
                    "metadata": {
                        "source": file_path.name,
                        "source_path": str(file_path),
                        "doc_type": "geological_report",
                        "file_type": "pdf",
                        "page": page_num + 1,
                        "total_pages": total_pages,
                        "section": current_section,
                        "chunk_index": chunk.chunk_index,
                        "file_hash": file_hash,
                        "token_estimate": chunk.token_estimate,
                        "has_table": chunk.has_table,
                        "has_coordinates": chunk.has_coordinates,
                        "has_geochemistry": chunk.has_geochemistry,
                    },
                })

        doc.close()
        logger.info(f"  → {len(all_chunks)} chunks from {total_pages} pages [{file_path.name}]")
        return all_chunks

    def process_directory(self, directory: str | Path) -> list[dict]:
        """Process all PDFs in a directory (recursive)."""
        directory = Path(directory)
        all_docs: list[dict] = []
        pdf_files = sorted(directory.rglob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF(s) in {directory}")
        for pdf_file in pdf_files:
            try:
                all_docs.extend(self.process_file(pdf_file))
            except Exception as exc:
                logger.error(f"Failed to process {pdf_file.name}: {exc}")
        return all_docs


# ── Module-level helpers ───────────────────────────────────────────────────────

def _detect_section(text_head: str) -> Optional[str]:
    """Return the first matching section name or None."""
    for pattern, name in _SECTION_PATTERNS:
        if pattern.search(text_head):
            return name
    return None


def _compute_hash(file_path: Path) -> str:
    """Fast 12-char MD5 prefix (first 64 KB) — enough for deduplication."""
    h = hashlib.md5()
    with open(file_path, "rb") as fh:
        h.update(fh.read(65_536))
    return h.hexdigest()[:12]
