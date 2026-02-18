"""
Embedder — encodes chunks with sentence-transformers and stores in ChromaDB.

Design decisions:
  - Model: all-MiniLM-L6-v2 (22M params, ~80MB, CPU-friendly, no GPU needed)
  - DB: ChromaDB persistent local storage — no server, no Docker
  - Similarity: cosine (stored as HNSW index inside ChromaDB)
  - Idempotent: re-ingesting the same file skips already-indexed chunks
"""

from __future__ import annotations
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from app.ingestion.chunker import Chunk

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_COLLECTION = "documents"
DEFAULT_DB_PATH    = "./data/chromadb"
DEFAULT_MODEL      = "all-MiniLM-L6-v2"
BATCH_SIZE         = 64


# ---------------------------------------------------------------------------
# EmbedderClient
# ---------------------------------------------------------------------------

class EmbedderClient:
    """
    Embeds and stores chunks in ChromaDB.
    Also handles querying by semantic similarity.

    Usage:
        client = EmbedderClient()
        client.add_chunks(chunks)
        results = client.query("what is multi-head attention?", n_results=5)
    """

    def __init__(
        self,
        db_path: str        = DEFAULT_DB_PATH,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str     = DEFAULT_MODEL,
    ):
        self.db_path         = db_path
        self.collection_name = collection_name
        self.model_name      = model_name

        # Create DB directory if needed
        Path(db_path).mkdir(parents=True, exist_ok=True)

        # ChromaDB — persistent, fully local, no server needed
        self.chroma = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )

        # cosine similarity via HNSW index
        self.collection = self.chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Embedding model — downloads once (~80MB), cached in ~/.cache
        print(f"  Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        dim = self.model.get_sentence_embedding_dimension()
        print(f"  Model ready — embedding dim: {dim}")

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[Chunk],
        show_progress: bool = True,
    ) -> int:
        """
        Embed and store chunks. Skips chunks already in the index (by chunk_id).

        Returns:
            Number of NEW chunks added.
        """
        if not chunks:
            return 0

        # Check which chunk_ids already exist
        existing = set(self._existing_ids([c.chunk_id for c in chunks]))
        new_chunks = [c for c in chunks if c.chunk_id not in existing]

        if not new_chunks:
            print(f"  All {len(chunks)} chunks already indexed — skipping.")
            return 0

        if existing:
            print(f"  Skipping {len(existing)} already-indexed chunks.")

        # Embed in batches
        added = 0
        batches = [
            new_chunks[i : i + BATCH_SIZE]
            for i in range(0, len(new_chunks), BATCH_SIZE)
        ]

        for batch in tqdm(batches, desc="  Embedding", unit="batch",
                          disable=not show_progress):
            texts      = [c.text       for c in batch]
            ids        = [c.chunk_id   for c in batch]
            metadatas  = [c.to_metadata() for c in batch]

            # normalize_embeddings=True means dot product == cosine similarity
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).tolist()

            self.collection.add(
                ids        = ids,
                embeddings = embeddings,
                documents  = texts,
                metadatas  = metadatas,
            )
            added += len(batch)

        return added

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def query(
        self,
        question: str,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Find the most semantically similar chunks for a question.

        Args:
            question:  Natural language query.
            n_results: How many chunks to return.
            where:     Optional metadata filter e.g. {"doc_id": "paper_abc"}

        Returns:
            List of result dicts, sorted by similarity score descending:
            [{"chunk_id", "text", "metadata", "score"}, ...]
            Score is 0.0–1.0 where 1.0 = identical.
        """
        total = self.collection.count()
        if total == 0:
            return []

        q_vec = self.model.encode(
            question,
            normalize_embeddings=True,
        ).tolist()

        kwargs: dict = dict(
            query_embeddings = [q_vec],
            n_results        = min(n_results, total),
            include          = ["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        raw = self.collection.query(**kwargs)

        results = []
        for i, chunk_id in enumerate(raw["ids"][0]):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to 0–1 similarity score
            distance = raw["distances"][0][i]
            score    = round(1.0 - distance / 2.0, 4)
            results.append({
                "chunk_id": chunk_id,
                "text":     raw["documents"][0][i],
                "metadata": raw["metadatas"][0][i],
                "score":    score,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def count(self) -> int:
        """Total chunks currently in the collection."""
        return self.collection.count()

    def list_documents(self) -> list[dict]:
        """Return one summary entry per unique doc_id."""
        if self.collection.count() == 0:
            return []

        all_meta = self.collection.get(include=["metadatas"])["metadatas"]
        docs: dict[str, dict] = {}
        for m in all_meta:
            did = m.get("doc_id", "unknown")
            if did not in docs:
                docs[did] = {
                    "doc_id":    did,
                    "doc_title": m.get("doc_title", ""),
                    "filename":  m.get("filename", ""),
                    "chunks":    0,
                }
            docs[did]["chunks"] += 1

        return list(docs.values())

    def delete_document(self, doc_id: str) -> int:
        """Remove all chunks for a doc_id. Returns count deleted."""
        ids = self.collection.get(
            where={"doc_id": doc_id}, include=[]
        )["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def _existing_ids(self, chunk_ids: list[str]) -> list[str]:
        """Return which of the given chunk_ids are already in the collection."""
        if not chunk_ids:
            return []
        try:
            return self.collection.get(ids=chunk_ids, include=[])["ids"]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Run directly: python -m app.ingestion.embedder <file.pdf> "<query>"
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from rich import print as rprint
    from app.ingestion.parser  import parse_pdf
    from app.ingestion.chunker import chunk_document

    if len(sys.argv) < 3:
        print('Usage: python -m app.ingestion.embedder <file.pdf> "<query>"')
        print('Example: python -m app.ingestion.embedder sample_docs/sample_paper.pdf "what is multi-head attention?"')
        sys.exit(1)

    pdf_path = sys.argv[1]
    query    = sys.argv[2]

    # ── Ingest ──────────────────────────────────────────────────────────────
    rprint(f"\n[bold]Step 1 — Parse + Chunk[/bold]")
    parsed = parse_pdf(pdf_path)
    chunks = chunk_document(parsed)
    rprint(f"  {len(parsed.sections)} sections → {len(chunks)} chunks")

    rprint(f"\n[bold]Step 2 — Embed + Store[/bold]")
    client = EmbedderClient()
    added  = client.add_chunks(chunks)
    rprint(f"  Added: [green]{added}[/green] new chunks")
    rprint(f"  Total in DB: [green]{client.count()}[/green]")

    # ── Query ───────────────────────────────────────────────────────────────
    rprint(f"\n[bold]Step 3 — Query[/bold]")
    rprint(f"  Question: [italic]{query}[/italic]\n")

    results = client.query(query, n_results=5)

    for i, r in enumerate(results, 1):
        m     = r["metadata"]
        color = "green" if r["score"] > 0.5 else "yellow" if r["score"] > 0.3 else "red"
        rprint(f"  [bold]#{i}[/bold]  [{color}]score={r['score']}[/{color}]")
        rprint(f"       [cyan]{m.get('citation', '')}[/cyan]")
        rprint(f"       {r['text'][:150].replace(chr(10), ' ')}…\n")
