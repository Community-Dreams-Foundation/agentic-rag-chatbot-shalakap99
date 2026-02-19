"""
Hybrid Retrieval — BM25 + Dense vector search fused via Reciprocal Rank Fusion.

Why this matters:
  - Dense search finds semantically similar chunks (good for concepts)
  - BM25 finds exact keyword matches (good for names, numbers, specific terms)
  - RRF combines both ranked lists without needing score normalization

RRF formula: score(d) = Σ 1 / (k + rank(d))
  where k=60 is a constant that reduces impact of very high rankings.
"""

from __future__ import annotations
from rank_bm25 import BM25Okapi
from app.ingestion.embedder import EmbedderClient


def hybrid_search(
    query: str,
    client: EmbedderClient,
    n_results: int = 5,
    where: dict | None = None,
    rrf_k: int = 60,
) -> list[dict]:
    """
    Hybrid BM25 + dense retrieval with Reciprocal Rank Fusion.

    Args:
        query:     Natural language query.
        client:    EmbedderClient with indexed documents.
        n_results: Final number of results to return.
        where:     Optional ChromaDB metadata filter.
        rrf_k:     RRF constant (default 60, standard value).

    Returns:
        Ranked list of result dicts: {chunk_id, text, metadata, score}
        Score is the RRF fusion score (higher = more relevant).
    """
    fetch_n = max(n_results * 4, 20)   # fetch more candidates for better coverage

    # ── 1. Dense retrieval ────────────────────────────────────────────────
    dense_results = client.query(query, n_results=fetch_n, where=where)
    if not dense_results:
        return []

    # ── 2. BM25 retrieval over the same candidate pool ───────────────────
    # Build BM25 index on the fly from the dense candidates.
    # This keeps BM25 scoped to the same document filter as dense search.
    corpus  = [r["text"] for r in dense_results]
    tokenized = [doc.lower().split() for doc in corpus]
    bm25    = BM25Okapi(tokenized)

    query_tokens = query.lower().split()
    bm25_scores  = bm25.get_scores(query_tokens)

    # Rank BM25 results (highest score = rank 0)
    bm25_ranking = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )

    # ── 3. Reciprocal Rank Fusion ─────────────────────────────────────────
    rrf_scores: dict[str, float] = {}

    # Dense ranks
    for rank, result in enumerate(dense_results):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

    # BM25 ranks
    for rank, idx in enumerate(bm25_ranking):
        cid = dense_results[idx]["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

    # ── 4. Sort by RRF score and return top-n ─────────────────────────────
    chunk_map = {r["chunk_id"]: r for r in dense_results}
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for chunk_id, rrf_score in ranked[:n_results]:
        if chunk_id in chunk_map:
            entry = dict(chunk_map[chunk_id])
            entry["score"] = round(rrf_score, 4)
            entry["retrieval_method"] = "hybrid_rrf"
            results.append(entry)

    return results
