"""
Ingest CLI — one command to parse, chunk, embed, and query.

Usage:
    python -m app.ingestion.ingest --pdf sample_docs/sample_paper.pdf
    python -m app.ingestion.ingest --query "what is attention?"
    python -m app.ingestion.ingest --list
    python -m app.ingestion.ingest --delete sample_paper
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from rich import print as rprint
from rich.table import Table
from rich.console import Console

from app.ingestion.parser   import parse_pdf
from app.ingestion.chunker  import chunk_document
from app.ingestion.embedder import EmbedderClient

console = Console()


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def ingest(pdf_path: str, client: EmbedderClient) -> dict:
    """
    Full parse → chunk → embed pipeline for one PDF.
    Returns a summary dict (used by make sanity later).
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")

    rprint(f"\n[bold]Ingesting:[/bold] {path.name}")

    with console.status("Parsing PDF…"):
        parsed = parse_pdf(path)
    rprint(f"  [green]✓[/green] Parsed   — {len(parsed.sections)} sections, "
           f"{parsed.num_pages} pages")
    rprint(f"  [dim]Title: {parsed.title[:70]}[/dim]")

    with console.status("Chunking…"):
        chunks = chunk_document(parsed)
    rprint(f"  [green]✓[/green] Chunked  — {len(chunks)} chunks")

    added = client.add_chunks(chunks, show_progress=True)
    rprint(f"  [green]✓[/green] Indexed  — {added} new chunks added "
           f"(total in DB: {client.count()})")

    return {
        "doc_id":    parsed.doc_id,
        "title":     parsed.title,
        "filename":  path.name,
        "pages":     parsed.num_pages,
        "sections":  len(parsed.sections),
        "chunks":    len(chunks),
        "new_added": added,
    }


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def query(question: str, client: EmbedderClient, n: int = 5,
          doc_id: str | None = None):
    """Run a semantic search query against the indexed documents."""
    rprint(f"\n[bold]Query:[/bold] {question}\n")

    where   = {"doc_id": doc_id} if doc_id else None
    results = client.query(question, n_results=n, where=where)

    if not results:
        rprint("[yellow]No results found. Have you ingested any documents?[/yellow]")
        return

    for i, r in enumerate(results, 1):
        m     = r["metadata"]
        score = r["score"]
        color = "green" if score > 0.5 else "yellow" if score > 0.3 else "red"
        rprint(f"[bold]#{i}[/bold]  [{color}]score={score}[/{color}]  "
               f"[cyan]{m.get('citation', '')}[/cyan]")
        rprint(f"  {r['text'][:220].replace(chr(10), ' ')}…\n")


# ---------------------------------------------------------------------------
# List documents
# ---------------------------------------------------------------------------

def list_documents(client: EmbedderClient):
    """Print a table of all indexed documents."""
    docs = client.list_documents()
    if not docs:
        rprint("[yellow]No documents indexed yet.[/yellow]")
        return

    t = Table(title="Indexed Documents", show_lines=True)
    t.add_column("doc_id",    style="cyan")
    t.add_column("title",     max_width=45)
    t.add_column("filename")
    t.add_column("chunks",    justify="right")

    for d in docs:
        t.add_row(
            d["doc_id"],
            d["doc_title"][:45],
            d["filename"],
            str(d["chunks"]),
        )
    console.print(t)
    rprint(f"\n[dim]Total chunks in index: {client.count()}[/dim]")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDFs and query the RAG knowledge base"
    )
    parser.add_argument("--pdf",    type=str, help="PDF file to ingest")
    parser.add_argument("--query",  type=str, help="Question to ask")
    parser.add_argument("--list",   action="store_true",
                        help="List all indexed documents")
    parser.add_argument("--delete", type=str, metavar="DOC_ID",
                        help="Remove a document from the index by doc_id")
    parser.add_argument("--n",      type=int, default=5,
                        help="Number of results to return (default: 5)")
    parser.add_argument("--doc",    type=str,
                        help="Filter query to a specific doc_id")
    parser.add_argument("--db",     type=str, default="./data/chromadb",
                        help="ChromaDB storage path (default: ./data/chromadb)")
    parser.add_argument("--output", type=str,
                        help="Save ingestion summary to a JSON file")
    args = parser.parse_args()

    client = EmbedderClient(db_path=args.db)

    if args.list:
        list_documents(client)

    if args.delete:
        n = client.delete_document(args.delete)
        rprint(f"[red]Deleted {n} chunks for doc_id '{args.delete}'[/red]")

    if args.pdf:
        summary = ingest(args.pdf, client)
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(summary, indent=2))
            rprint(f"\n[dim]Summary saved to {args.output}[/dim]")

    if args.query:
        query(args.query, client, n=args.n, doc_id=args.doc)


if __name__ == "__main__":
    main()
