"""
Memory Writer — maintains USER_MEMORY.md and COMPANY_MEMORY.md.

Memory is SELECTIVE:
  - Explicit preferences ("I prefer...", "I like...")
  - Role/identity statements ("I am a...", "I work as...")
  - NOT raw transcripts, NOT every question asked
  - NOT PII, secrets, or sensitive data
"""

from __future__ import annotations
import re
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

# Patterns that signal a high-signal user fact worth remembering
_PREFERENCE_PATTERNS = [
    re.compile(r"\bi\s+prefer\b(.+)", re.IGNORECASE),
    re.compile(r"\bi\s+like\b(.+)", re.IGNORECASE),
    re.compile(r"\bi\s+always\b(.+)", re.IGNORECASE),
    re.compile(r"\bi\s+want\b(.+)", re.IGNORECASE),
    re.compile(r"\bplease\s+(always|never|don't|do)\b(.+)", re.IGNORECASE),
]

_IDENTITY_PATTERNS = [
    re.compile(r"\bi\s+am\s+a[n]?\b(.+)", re.IGNORECASE),
    re.compile(r"\bi'm\s+a[n]?\b(.+)", re.IGNORECASE),
    re.compile(r"\bi\s+work\s+as\b(.+)", re.IGNORECASE),
    re.compile(r"\bmy\s+role\s+is\b(.+)", re.IGNORECASE),
    re.compile(r"\bi\s+work\s+(at|for|in)\b(.+)", re.IGNORECASE),
]


def extract_user_facts(messages: list[str]) -> dict[str, list[str]]:
    """
    Scan user messages for high-signal facts.
    Returns {"preferences": [...], "identity": [...]}
    Only stores facts — never raw questions or transcripts.
    """
    preferences: list[str] = []
    identity:    list[str] = []
    seen = set()   # deduplicate

    for msg in messages:
        msg = msg.strip()

        for pattern in _PREFERENCE_PATTERNS:
            m = pattern.search(msg)
            if m:
                fact = msg.strip().rstrip(".")
                if fact.lower() not in seen:
                    seen.add(fact.lower())
                    preferences.append(fact)

        for pattern in _IDENTITY_PATTERNS:
            m = pattern.search(msg)
            if m:
                fact = msg.strip().rstrip(".")
                if fact.lower() not in seen:
                    seen.add(fact.lower())
                    identity.append(fact)

    return {"preferences": preferences, "identity": identity}


def write_memory(
    user_questions: list[str],
    indexed_docs:   list[dict],
    user_id:        str = "default",
) -> None:
    """Update both memory files. Called after every ingestion and Q&A."""
    _write_user_memory(user_questions, user_id)
    _write_company_memory(indexed_docs)


def _write_user_memory(messages: list[str], user_id: str):
    now   = datetime.now().strftime("%Y-%m-%d %H:%M")
    facts = extract_user_facts(messages)
    topics = _infer_topics(messages)

    # Read existing memory to merge (don't overwrite previously stored facts)
    existing = USER_MEMORY_PATH.read_text() if USER_MEMORY_PATH.exists() else ""
    existing_prefs = _extract_existing_section(existing, "## Preferences")
    existing_identity = _extract_existing_section(existing, "## Identity")

    # Merge new facts with existing, deduplicated
    all_prefs    = _merge_unique(existing_prefs, facts["preferences"])
    all_identity = _merge_unique(existing_identity, facts["identity"])

    lines = [
        "# User Memory",
        f"\n_Last updated: {now}_\n",
        "## Identity",
    ]
    if all_identity:
        for f in all_identity:
            lines.append(f"- {f}")
    else:
        lines.append("- Not yet provided.")

    lines += ["", "## Preferences"]
    if all_prefs:
        for f in all_prefs:
            lines.append(f"- {f}")
    else:
        lines.append("- Not yet provided.")

    lines += ["", "## Topics of Interest"]
    if topics:
        for topic, count in topics:
            lines.append(f"- {topic} (×{count})")
    else:
        lines.append("- No topics inferred yet.")

    lines += [
        "",
        "## Note",
        "- Raw conversation transcripts are NOT stored.",
        "- Only high-signal facts (identity, preferences) are recorded.",
    ]

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


def _infer_topics(messages: list[str]) -> list[tuple[str, int]]:
    counts: Counter = Counter()
    for q in messages:
        for w in q.lower().split():
            w = w.strip("?.,!\"'()")
            if len(w) > 3 and w not in _STOPWORDS:
                counts[w] += 1
    return counts.most_common(8)


def _extract_existing_section(text: str, header: str) -> list[str]:
    """Pull bullet items from an existing memory section."""
    items = []
    in_section = False
    for line in text.splitlines():
        if line.strip() == header:
            in_section = True
            continue
        if in_section:
            if line.startswith("## "):
                break
            if line.startswith("- ") and "Not yet provided" not in line:
                items.append(line[2:].strip())
    return items


def _merge_unique(existing: list[str], new: list[str]) -> list[str]:
    """Merge two lists, deduplicating by lowercase comparison."""
    seen  = {x.lower() for x in existing}
    result = list(existing)
    for item in new:
        if item.lower() not in seen:
            seen.add(item.lower())
            result.append(item)
    return result
