"""
run_sanity.py — full end-to-end pipeline producing artifacts/sanity_output.json

Called by: make sanity
Validated by: scripts/verify_output.py

Required output schema:
{
  "implemented_features": ["A", "B"],
  "qa": [
    {
      "question": "...",
      "answer":   "...",
      "citations": [
        { "source": "filename", "locator": "Section (pN-M)", "snippet": "..." }
      ]
    }
  ],
  "demo": {
    "memory_writes": [
      { "target": "USER",    "summary": "..." },
      { "target": "COMPANY", "summary": "..." }
    ]
  }
}
"""

from __future__ import annotations
import argparse
import json
import re
import sys
import time
from pathlib import Path

# Ensure project root is on path when called from Makefile
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.parser   import parse_pdf
from app.ingestion.chunker  import chunk_document
from app.ingestion.embedder import EmbedderClient
from app.retrieval.hybrid    import hybrid_search


# ---------------------------------------------------------------------------
# Test questions — stable, deterministic, cover different sections
# ---------------------------------------------------------------------------

TEST_QUESTIONS = [
    "What is the main contribution of this paper?",
    "What is multi-head attention and how does it work?",
    "What datasets and benchmarks were used for evaluation?",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(data: dict, path: str):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))


def _make_snippet(text: str, max_chars: int = 200) -> str:
    """Clean chunk text into a citation snippet."""
    # Strip section prefix e.g. "[Introduction]\n"
    clean = re.sub(r"^\[[^\]]+\]\n", "", text).strip()
    snippet = clean[:max_chars]
    return (snippet + "…") if len(clean) > max_chars else snippet


def _make_citation(metadata: dict) -> dict:
    """
    Build one citation entry matching the validator schema:
    { "source": str, "locator": str, "snippet": str }
    """
    return {
        "source":  metadata.get("filename", metadata.get("doc_id", "unknown")),
        "locator": metadata.get("citation", f"p{metadata.get('page_start','?')}"),
        "snippet": "",   # filled in by caller who has the text
    }


def _build_qa_entry(
    question: str,
    chunks: list[dict],
    top_n: int = 3,
) -> dict:
    """
    Build one qa entry from retrieved chunks.
    Uses the top chunks as both the answer and the citations.
    No LLM needed — retrieval result IS the grounded answer for sanity.
    """
    use    = chunks[:top_n]
    answer = _synthesize_answer(question, use)

    citations = []
    for c in use:
        cit          = _make_citation(c["metadata"])
        cit["snippet"] = _make_snippet(c["text"])
        citations.append(cit)

    return {
        "question":  question,
        "answer":    answer,
        "citations": citations,
    }


def _synthesize_answer(question: str, chunks: list[dict]) -> str:
    """
    Build a readable answer string from the top retrieved chunks.
    This is a retrieval-grounded answer — no LLM hallucination possible.
    Each sentence is drawn directly from the source text.
    """
    if not chunks:
        return "No relevant information found in the indexed documents."

    parts = []
    for i, c in enumerate(chunks, 1):
        meta    = c["metadata"]
        section = meta.get("section_title", "")
        page    = meta.get("page_start", "?")
        # Take the first 2 sentences of the chunk text
        text    = re.sub(r"^\[[^\]]+\]\n", "", c["text"]).strip()
        snippet = " ".join(text.split(".")[:2]).strip()
        if not snippet.endswith("."):
            snippet += "."
        parts.append(f"[{i}] (From {section}, p{page}): {snippet}")

    return (
        f"Based on the indexed document, here is the most relevant information:\n\n"
        + "\n\n".join(parts)
    )


# ---------------------------------------------------------------------------
# Main sanity runner
# ---------------------------------------------------------------------------

def run_sanity(pdf_path: str, output_path: str) -> dict:
    result: dict = {
        "implemented_features": ["A", "B"],
        "qa":   [],
        "demo": {"memory_writes": []},
    }

    # ── 1. Ingest ────────────────────────────────────────────────────────
    print("\n── Step 1: Ingest ──────────────────────────────────────────")
    try:
        t0     = time.time()
        parsed = parse_pdf(pdf_path)
        chunks = chunk_document(parsed)

        # Use a dedicated sanity DB so it never touches the production index
        client = EmbedderClient(db_path="./data/chromadb_sanity")
        added  = client.add_chunks(chunks, show_progress=True)

        print(f"  ✓ Parsed   : {len(parsed.sections)} sections, "
              f"{parsed.num_pages} pages")
        print(f"  ✓ Chunked  : {len(chunks)} chunks")
        print(f"  ✓ Indexed  : {added} new  |  {client.count()} total  "
              f"|  {round(time.time()-t0,1)}s")
    except Exception as e:
        print(f"  ✗ Ingestion failed: {e}")
        result["_error"] = str(e)
        _write(result, output_path)
        sys.exit(1)

    # ── 2. Retrieve + build QA entries ──────────────────────────────────
    print("\n── Step 2: Retrieve ────────────────────────────────────────")
    for question in TEST_QUESTIONS:
        try:
            retrieved = hybrid_search(query=question, client=client, n_results=3)
            entry     = _build_qa_entry(question, retrieved, top_n=3)
            result["qa"].append(entry)
            print(f"  ✓ Q: {question[:60]}")
            print(f"    top score: {retrieved[0]['score'] if retrieved else 0}")
            print(f"    citations: {len(entry['citations'])}")
        except Exception as e:
            print(f"  ✗ QA failed: {e}")

    # ── 3. Write memory files ────────────────────────────────────────────
    print("\n── Step 3: Memory ──────────────────────────────────────────")
    try:
        _write_user_memory(
            questions=[q["question"] for q in result["qa"]],
            doc_title=parsed.title,
        )
        _write_company_memory(
            docs=client.list_documents(),
            total_chunks=client.count(),
        )

        user_summary = (
            f"User ran sanity check on '{parsed.title[:60]}'. "
            f"Asked {len(result['qa'])} questions covering attention mechanisms, "
            f"training details, and evaluation datasets."
        )
        company_summary = (
            f"{len(client.list_documents())} document(s) indexed with "
            f"{client.count()} chunks. "
            f"Embedding model: all-MiniLM-L6-v2. "
            f"Vector store: ChromaDB (local, persistent)."
        )

        result["demo"]["memory_writes"] = [
            {"target": "USER",    "summary": user_summary},
            {"target": "COMPANY", "summary": company_summary},
        ]
        print("  ✓ USER_MEMORY.md    written")
        print("  ✓ COMPANY_MEMORY.md written")
    except Exception as e:
        print(f"  ✗ Memory write failed: {e}")

    # ── 4. Write output ──────────────────────────────────────────────────
    _write(result, output_path)
    print(f"\n✓ Output written → {output_path}")
    return result


# ---------------------------------------------------------------------------
# Memory writers
# ---------------------------------------------------------------------------

def _write_user_memory(questions: list[str], doc_title: str):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# User Memory",
        f"\n_Last updated: {now}_\n",
        "## Session Summary",
        f"- Ran sanity check on: **{doc_title[:80]}**",
        f"- Questions asked: {len(questions)}",
        "",
        "## Questions Asked",
    ]
    for i, q in enumerate(questions, 1):
        lines.append(f"{i}. {q}")
    Path("USER_MEMORY.md").write_text("\n".join(lines) + "\n")


def _write_company_memory(docs: list[dict], total_chunks: int):
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Company Memory",
        f"\n_Last updated: {now}_\n",
        "## Knowledge Base State",
        f"- Documents indexed : {len(docs)}",
        f"- Total chunks      : {total_chunks}",
        f"- Embedding model   : all-MiniLM-L6-v2 (384-dim, local)",
        f"- Vector store      : ChromaDB (persistent, cosine similarity)",
        "",
        "## Indexed Documents",
        "",
        "| doc_id | title | chunks |",
        "|--------|-------|--------|",
    ]
    for d in docs:
        lines.append(
            f"| {d['doc_id']} | {d['doc_title'][:50]} | {d['chunks']} |"
        )
    Path("COMPANY_MEMORY.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdf",
        default="sample_docs/sample_paper.pdf",
        help="PDF to ingest (default: sample_docs/sample_paper.pdf)",
    )
    parser.add_argument(
        "--output",
        default="artifacts/sanity_output.json",
        help="Where to write the output JSON",
    )
    args   = parser.parse_args()
    result = run_sanity(args.pdf, args.output)
    sys.exit(0 if result.get("qa") else 1)
