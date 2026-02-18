"""
PDF Parser — extracts text with section awareness.

Strategy:
  - Uses PyMuPDF (fitz) for layout-preserving extraction
  - Detects section headings via font size heuristics
  - Post-processes to merge noise sections into their neighbors
  - Falls back to page-level splitting if no real headings found
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # pymupdf


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Section:
    title: str
    text: str
    page_start: int
    page_end: int
    section_index: int

@dataclass
class ParsedDocument:
    doc_id: str
    filename: str
    title: str
    sections: list[Section] = field(default_factory=list)
    raw_text: str = ""
    num_pages: int = 0


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

_KNOWN_HEADINGS = {
    "abstract", "introduction", "background", "related work",
    "methodology", "method", "methods", "approach", "model",
    "experiments", "experiment", "experimental setup", "results",
    "evaluation", "discussion", "conclusion", "conclusions",
    "future work", "acknowledgments", "acknowledgements", "references",
    "appendix", "model architecture", "training", "encoder", "decoder",
    "attention", "why self-attention", "positional encoding",
    "position-wise feed-forward networks", "embeddings and softmax",
    "multi-head attention", "scaled dot-product attention",
    "applications of attention in our model", "training data and batching",
    "hardware and schedule", "optimizer", "residual dropout",
    "label smoothing", "machine translation", "model variations",
    "english constituency parsing", "attention visualizations",
}

# Patterns that are definitely NOT headings regardless of font size
_NOISE_RE = re.compile(
    r"(arxiv|doi|http|www\.|@|\d{4}\.\d{4}|"   # arxiv IDs, URLs, emails
    r"^\s*\d+\s*$|"                              # bare page numbers
    r"where the projections|"                    # formula continuation lines
    r"we employ|"                                # sentence fragments
    r"^(the|a|an)\s+\w+\s*$)",                  # "The Law" style fragments
    re.IGNORECASE,
)

def _is_heading(text: str, size: float, body_size: float, page: int) -> bool:
    clean = text.strip()

    # Basic guards
    if not clean or len(clean) > 80 or len(clean) < 3:
        return False

    # Never treat noise patterns as headings
    if _NOISE_RE.search(clean):
        return False

    # Known exact heading names (case-insensitive) — always a heading
    if clean.lower() in _KNOWN_HEADINGS:
        return True

    # Numbered section pattern: "3.", "3.1", "3.1 Title"
    if re.match(r"^\d+(\.\d+)?\s+[A-Z]", clean):
        return True

    # Font-size signal — only trust it after page 1
    # (page 1 has author names at heading-size fonts)
    if page > 1 and size >= body_size * 1.25:
        # Still must look like a title: starts with capital, no sentence punctuation
        if re.match(r"^[A-Z][A-Za-z0-9\s\-:]{2,60}$", clean):
            if not clean.endswith((",", ";", ":", ".", "?")):
                return True

    return False


def _estimate_body_font_size(doc: fitz.Document) -> float:
    from collections import Counter
    sizes: Counter = Counter()
    for page in doc[:5]:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sz = round(span["size"])
                    sizes[sz] += len(span["text"])
    return float(sizes.most_common(1)[0][0]) if sizes else 11.0


# ---------------------------------------------------------------------------
# Post-processing: remove noise sections
# ---------------------------------------------------------------------------

# Minimum number of words for a section to be kept as its own entry.
# Sections with fewer words get merged into the previous section.
_MIN_SECTION_WORDS = 30

def _merge_noise_sections(sections: list[Section]) -> list[Section]:
    """
    Merge sections that are too short (noise) into the previous section.
    Re-indexes the kept sections cleanly.
    """
    if not sections:
        return sections

    merged: list[Section] = []
    for sec in sections:
        word_count = len(sec.text.split())
        if word_count < _MIN_SECTION_WORDS and merged:
            # Append this section's text to the previous one
            prev = merged[-1]
            merged[-1] = Section(
                title=prev.title,
                text=prev.text + "\n" + sec.text,
                page_start=prev.page_start,
                page_end=sec.page_end,
                section_index=prev.section_index,
            )
        else:
            merged.append(sec)

    # Re-index cleanly
    for i, sec in enumerate(merged):
        sec.section_index = i

    return merged


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_pdf(pdf_path: str | Path) -> ParsedDocument:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    doc       = fitz.open(str(path))
    body_size = _estimate_body_font_size(doc)
    doc_id    = path.stem.replace(" ", "_").lower()

    # ── Pass 1: extract all lines with font metadata ──────────────────────
    raw_blocks: list[dict] = []
    full_text_parts: list[str] = []

    for page_num, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                line_text = ""
                max_size  = 0.0
                for span in line.get("spans", []):
                    line_text += span["text"]
                    if span["size"] > max_size:
                        max_size = span["size"]
                line_text = line_text.strip()
                if line_text:
                    raw_blocks.append({
                        "text": line_text,
                        "size": max_size,
                        "page": page_num,
                    })
                    full_text_parts.append(line_text)

    raw_text = "\n".join(full_text_parts)

    # ── Pass 2: group lines into sections ─────────────────────────────────
    sections: list[Section] = []
    current_title      = "Preamble"
    current_page_start = 1
    current_lines: list[str] = []

    def _flush(title, page_start, page_end, lines, idx):
        text = "\n".join(lines).strip()
        if text:
            sections.append(Section(
                title=title,
                text=text,
                page_start=page_start,
                page_end=page_end,
                section_index=idx,
            ))

    for blk in raw_blocks:
        if _is_heading(blk["text"], blk["size"], body_size, blk["page"]):
            _flush(current_title, current_page_start,
                   blk["page"], current_lines, len(sections))
            current_title      = blk["text"].strip()
            current_page_start = blk["page"]
            current_lines      = []
        else:
            current_lines.append(blk["text"])

    _flush(current_title, current_page_start,
           doc.page_count, current_lines, len(sections))

    # ── Pass 3: merge noise sections ──────────────────────────────────────
    sections = _merge_noise_sections(sections)

    # ── Fallback: page-level if fewer than 3 sections detected ───────────
    if len(sections) <= 2:
        sections = [
            Section(
                title=f"Page {i + 1}",
                text=doc[i].get_text("text").strip(),
                page_start=i + 1,
                page_end=i + 1,
                section_index=i,
            )
            for i in range(doc.page_count)
            if doc[i].get_text("text").strip()
        ]

    title     = _extract_title(doc)
    num_pages = doc.page_count
    doc.close()

    return ParsedDocument(
        doc_id=doc_id,
        filename=path.name,
        title=title,
        sections=sections,
        raw_text=raw_text,
        num_pages=num_pages,
    )


def _extract_title(doc: fitz.Document) -> str:
    try:
        spans = []
        for block in doc[0].get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span["text"].strip():
                        spans.append((span["size"], span["text"].strip()))
        if spans:
            spans.sort(reverse=True)
            return spans[0][1][:120]
    except Exception:
        pass
    return "Unknown Title"


# ---------------------------------------------------------------------------
# Run directly: python -m app.ingestion.parser <file.pdf>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from rich import print as rprint

    if len(sys.argv) < 2:
        print("Usage: python -m app.ingestion.parser <path/to/paper.pdf>")
        sys.exit(1)

    result = parse_pdf(sys.argv[1])
    rprint(f"\n[bold green]Title   :[/bold green] {result.title}")
    rprint(f"[bold]Pages   :[/bold] {result.num_pages}")
    rprint(f"[bold]Sections:[/bold] {len(result.sections)}\n")
    for s in result.sections:
        word_count = len(s.text.split())
        preview    = s.text[:80].replace("\n", " ")
        rprint(
            f"  [{s.section_index:02d}] [cyan]{s.title:<45}[/cyan] "
            f"p{s.page_start}–{s.page_end}  ({word_count}w)  {preview}…"
        )
