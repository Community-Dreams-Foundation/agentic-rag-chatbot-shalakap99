"""
Chunker — converts ParsedDocument sections into embeddable Chunk objects.

Strategy:
  - If a section fits within max_chars: emit as one chunk
  - If a section is too long: split on sentence boundaries with overlap
  - Every chunk is prefixed with its section title so context is never lost
  - Every chunk carries full metadata for citations
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Iterator

from app.ingestion.parser import ParsedDocument, Section


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id:      str   # "{doc_id}__sec{section_index}__chunk{chunk_index}"
    doc_id:        str
    doc_title:     str
    filename:      str
    section_title: str
    section_index: int
    chunk_index:   int   # position within its section (0-based)
    text:          str
    page_start:    int
    page_end:      int
    char_count:    int = 0

    def __post_init__(self):
        self.char_count = len(self.text)

    @property
    def citation(self) -> str:
        """Human-readable citation used in LLM prompts and UI."""
        return (
            f"{self.doc_title} › {self.section_title} "
            f"(p{self.page_start}–{self.page_end})"
        )

    def to_metadata(self) -> dict:
        """
        Flat dict for ChromaDB metadata storage.
        All values must be str, int, float, or bool — no nested objects.
        """
        return {
            "doc_id":        self.doc_id,
            "doc_title":     self.doc_title[:256],
            "filename":      self.filename,
            "section_title": self.section_title,
            "section_index": self.section_index,
            "chunk_index":   self.chunk_index,
            "page_start":    self.page_start,
            "page_end":      self.page_end,
            "citation":      self.citation[:512],
        }


# ---------------------------------------------------------------------------
# Sentence splitter — no NLTK dependency
# ---------------------------------------------------------------------------

# Split after . ! ? when followed by whitespace + capital letter
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"(\[])")

def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def chunk_document(
    doc: ParsedDocument,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> list[Chunk]:
    """
    Convert a ParsedDocument into a flat list of Chunk objects.

    Args:
        doc:           Output of parse_pdf()
        max_chars:     Max characters per chunk (~300 tokens at 4 chars/token)
        overlap_chars: Characters of overlap carried into the next chunk
                       (preserves context across chunk boundaries)

    Returns:
        Flat list of Chunk objects, ordered by section then position.
    """
    chunks: list[Chunk] = []
    for section in doc.sections:
        for chunk in _chunk_section(
            section=section,
            doc_id=doc.doc_id,
            doc_title=doc.title,
            filename=doc.filename,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        ):
            chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Internal: chunk one section
# ---------------------------------------------------------------------------

def _chunk_section(
    section: Section,
    doc_id: str,
    doc_title: str,
    filename: str,
    max_chars: int,
    overlap_chars: int,
) -> Iterator[Chunk]:
    """Yield one or more chunks from a single section."""
    text = section.text.strip()
    if not text:
        return

    # Prefix every chunk with its section title so the LLM always
    # knows which part of the paper it is reading
    prefix = f"[{section.title}]\n"

    # ── Case 1: entire section fits in one chunk ──────────────────────────
    if len(prefix) + len(text) <= max_chars:
        yield Chunk(
            chunk_id      = f"{doc_id}__sec{section.section_index}__chunk0",
            doc_id        = doc_id,
            doc_title     = doc_title,
            filename      = filename,
            section_title = section.title,
            section_index = section.section_index,
            chunk_index   = 0,
            text          = prefix + text,
            page_start    = section.page_start,
            page_end      = section.page_end,
        )
        return

    # ── Case 2: section too long — split on sentence boundaries ──────────
    sentences = _split_sentences(text)
    if not sentences:
        sentences = [text]   # no sentence boundaries found — treat as one

    chunk_index:    int  = 0
    current_sents:  list[str] = []
    current_len:    int  = len(prefix)
    overlap_buffer: str  = ""

    for sent in sentences:
        sent_len = len(sent) + 1   # +1 for the space when joining

        if current_len + sent_len > max_chars and current_sents:
            # ── Emit current chunk ────────────────────────────────────────
            chunk_text = (prefix + overlap_buffer + " ".join(current_sents)).strip()
            yield Chunk(
                chunk_id      = f"{doc_id}__sec{section.section_index}__chunk{chunk_index}",
                doc_id        = doc_id,
                doc_title     = doc_title,
                filename      = filename,
                section_title = section.title,
                section_index = section.section_index,
                chunk_index   = chunk_index,
                text          = chunk_text,
                page_start    = section.page_start,
                page_end      = section.page_end,
            )
            chunk_index += 1

            # ── Build overlap from the tail of the current chunk ──────────
            # Walk backwards through current sentences until we fill
            # overlap_chars — this carries context into the next chunk
            overlap_parts: list[str] = []
            overlap_len = 0
            for part in reversed(current_sents):
                if overlap_len + len(part) > overlap_chars:
                    break
                overlap_parts.insert(0, part)
                overlap_len += len(part) + 1
            overlap_buffer = " ".join(overlap_parts) + " " if overlap_parts else ""

            current_sents = [sent]
            current_len   = len(prefix) + len(overlap_buffer) + sent_len
        else:
            current_sents.append(sent)
            current_len += sent_len

    # ── Flush the last remaining sentences ───────────────────────────────
    if current_sents:
        chunk_text = (prefix + overlap_buffer + " ".join(current_sents)).strip()
        yield Chunk(
            chunk_id      = f"{doc_id}__sec{section.section_index}__chunk{chunk_index}",
            doc_id        = doc_id,
            doc_title     = doc_title,
            filename      = filename,
            section_title = section.title,
            section_index = section.section_index,
            chunk_index   = chunk_index,
            text          = chunk_text,
            page_start    = section.page_start,
            page_end      = section.page_end,
        )


# ---------------------------------------------------------------------------
# Run directly: python -m app.ingestion.chunker <file.pdf>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from collections import Counter
    from rich import print as rprint
    from app.ingestion.parser import parse_pdf

    if len(sys.argv) < 2:
        print("Usage: python -m app.ingestion.chunker <path/to/paper.pdf>")
        sys.exit(1)

    parsed = parse_pdf(sys.argv[1])
    chunks = chunk_document(parsed)

    # Summary by section
    sec_counts: Counter = Counter(c.section_title for c in chunks)

    rprint(f"\n[bold green]Total chunks:[/bold green] {len(chunks)}")
    rprint(f"[bold]From sections:[/bold] {len(parsed.sections)}\n")
    rprint("[bold]Chunks per section:[/bold]")
    for sec, count in sec_counts.most_common():
        rprint(f"  {count:2d}  [cyan]{sec}[/cyan]")

    rprint(f"\n[bold]First 3 chunk previews:[/bold]")
    for c in chunks[:3]:
        rprint(f"\n  [bold cyan]{c.chunk_id}[/bold cyan]")
        rprint(f"  Citation : {c.citation}")
        rprint(f"  Chars    : {c.char_count}")
        rprint(f"  Preview  : {c.text[:120].replace(chr(10), ' ')}…")
