"""
Memory Writer — updates USER_MEMORY.md and COMPANY_MEMORY.md after
each ingestion and Q&A session. Required by submission rules.
"""

from __future__ import annotations
from collections import Counter
from datetime import datetime
from pathlib import Path

USER_MEMORY_PATH    = Path("USER_MEMORY.md")
COMPANY_MEMORY_PATH = Path("COMPANY_MEMORY.md")

_STOPWORDS = {
    "what", "how", "why", "when", "where", "who", "which", "is", "are",
    "was", "were", "the", "a", "an", "of", "in", "to", "for", "on",
    "and", "or", "but", "with", "this", "that", "do", "does", "did",
    "can", "could", "would", "should", "will", "about", "from", "by",
    "it", "its", "be", "been", "have", "has", "had", "not", "paper",
    "document", "section", "used", "using", "use",
}


def write_memory(
    user_questions: list[str],
    indexed_docs:   list[dict],
    user_id:        str = "default",
) -> None:
    """Update both memory files. Called after every ingestion and Q&A."""
    _write_user_memory(user_questions, user_id)
    _write_company_memory(indexed_docs)


def _write_user_memory(questions: list[str], user_id: str):
    now    = datetime.now().strftime("%Y-%m-%d %H:%M")
    topics = _infer_topics(questions)

    lines = [
        "# User Memory",
        f"\n_Last updated: {now}_\n",
        "## Profile",
        f"- User ID : {user_id}",
        f"- Questions asked this session: {len(questions)}",
        "",
        "## Inferred Topics of Interest",
    ]
    if topics:
        for topic, count in topics:
            lines.append(f"- {topic} (×{count})")
    else:
        lines.append("- No topics inferred yet.")

    lines += ["", "## Question History _(most recent first)_", ""]
    for i, q in enumerate(reversed(questions), 1):
        lines.append(f"{i}. {q}")

    USER_MEMORY_PATH.write_text("\n".join(lines) + "\n")


def _write_company_memory(indexed_docs: list[dict]):
    now          = datetime.now().strftime("%Y-%m-%d %H:%M")
    total_chunks = sum(d.get("chunks", 0) for d in indexed_docs)

    lines = [
        "# Company Memory",
        f"\n_Last updated: {now}_\n",
        "## Knowledge Base",
        f"- Documents : {len(indexed_docs)}",
        f"- Chunks    : {total_chunks}",
        f"- Embeddings: all-MiniLM-L6-v2 (384-dim, local)",
        f"- Store     : ChromaDB (persistent, cosine similarity)",
        "",
        "## Indexed Documents",
        "",
        "| # | doc_id | title | chunks |",
        "|---|--------|-------|--------|",
    ]
    for i, d in enumerate(indexed_docs, 1):
        lines.append(
            f"| {i} | {d['doc_id']} "
            f"| {d['doc_title'][:50]} "
            f"| {d['chunks']} |"
        )

    COMPANY_MEMORY_PATH.write_text("\n".join(lines) + "\n")


def _infer_topics(questions: list[str]) -> list[tuple[str, int]]:
    counts: Counter = Counter()
    for q in questions:
        for w in q.lower().split():
            w = w.strip("?.,!\"'()")
            if len(w) > 3 and w not in _STOPWORDS:
                counts[w] += 1
    return counts.most_common(10)
