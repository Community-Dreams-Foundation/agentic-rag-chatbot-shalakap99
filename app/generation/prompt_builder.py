"""
Prompt Builder — constructs grounded RAG prompts with citation instructions.

Key design decisions:
  - System prompt explicitly forbids answering outside the provided context
  - Each source chunk gets a reference label [1], [2], [3]
  - LLM is told to cite inline: "Transformers use attention [1][3]"
  - Graceful failure path: if no relevant context, say so — never hallucinate
  - Conversation history included for multi-turn support
"""

from __future__ import annotations
import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# System prompt — the most important part of the RAG pipeline
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a precise research assistant. Answer questions using ONLY the document excerpts provided below.

Rules you must follow without exception:
1. Every claim must come from the provided excerpts. Do not use outside knowledge.
2. You MUST cite every sentence or bullet point with [1] or [2] style brackets.
   No sentence is allowed without a citation. This is mandatory, not optional.
3. If the same point appears in multiple sources, cite all: [1][3].
4. If the answer is not in the excerpts, say exactly:
   "I could not find relevant information in the uploaded documents."
5. End your answer with a "Sources:" section listing only what you cited.

Example of correct response format:
- The model replaces recurrence with self-attention mechanisms [1].
- Training is faster due to full parallelization [2].
- State-of-the-art BLEU scores were achieved on WMT 2014 [3].

Sources:
[1] Paper › Introduction (p2)
[2] Paper › Training (p7)
[3] Paper › Results (p8)

You will be penalized for any sentence missing a citation bracket."""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PromptResult:
    messages:           list[dict]   # ready to pass to OllamaClient.chat()
    sources:            list[dict]   # chunks used, in citation order
    context_char_count: int


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_grounded_prompt(
    question:             str,
    retrieved_chunks:     list[dict],
    conversation_history: list[dict] | None = None,
    min_score_threshold:  float = 0.30,
) -> PromptResult:
    """
    Build a messages list for a grounded RAG answer.

    Args:
        question:             The user's current question.
        retrieved_chunks:     Ranked chunks from EmbedderClient.query().
        conversation_history: Prior turns as [{"role":..., "content":...}].
        min_score_threshold:  Drop chunks below this similarity score.

    Returns:
        PromptResult with .messages ready for OllamaClient.chat().
    """
    # Filter out low-relevance chunks
    usable = [
        c for c in retrieved_chunks
        if c.get("score", 1.0) >= min_score_threshold
    ]

    if not usable:
        context_block = "[No relevant document excerpts found for this question.]"
        sources       = []
    else:
        context_block, sources = _build_context_block(usable)

    user_content = (
        f"Document excerpts:\n"
        f"──────────────────────────────────────\n"
        f"{context_block}\n"
        f"──────────────────────────────────────\n\n"
        f"Question: {question}"
    )

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject prior conversation turns (skip system messages)
    if conversation_history:
        for turn in conversation_history:
            if turn.get("role") in ("user", "assistant"):
                messages.append(turn)

    messages.append({"role": "user", "content": user_content})

    return PromptResult(
        messages=messages,
        sources=sources,
        context_char_count=len(context_block),
    )


def _build_context_block(chunks: list[dict]) -> tuple[str, list[dict]]:
    """
    Format retrieved chunks into a numbered context block.
    Returns (context_string, sources_list).
    """
    parts:   list[str]  = []
    sources: list[dict] = []

    for i, chunk in enumerate(chunks, start=1):
        meta     = chunk.get("metadata", {})
        citation = meta.get("citation", chunk.get("chunk_id", f"Source {i}"))
        text     = chunk.get("text", "").strip()

        parts.append(f"[{i}] {citation}\n{text}")
        sources.append({
            "ref":      i,
            "citation": citation,
            "chunk_id": chunk.get("chunk_id", ""),
            "score":    chunk.get("score", 0.0),
            "text":     text,
            # Fields needed by sanity_output.json citation schema
            "source":   meta.get("filename", meta.get("doc_id", "unknown")),
            "locator":  citation,
            "snippet":  text[:200],
        })

    return "\n\n".join(parts), sources


# ---------------------------------------------------------------------------
# Citation extraction — parse [N] refs from LLM response text
# ---------------------------------------------------------------------------

_CITE_RE = re.compile(r"\[(\d+)\]")

def extract_cited_sources(
    response_text: str,
    sources: list[dict],
) -> list[dict]:
    """
    Return only the sources that were actually cited in the response.
    Parses [1], [2] patterns from the LLM answer text.
    """
    cited_nums = {int(n) for n in _CITE_RE.findall(response_text)}
    return [s for s in sources if s["ref"] in cited_nums]


# ---------------------------------------------------------------------------
# Run directly: python -m app.generation.prompt_builder
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from rich import print as rprint

    # Simulate two retrieved chunks
    fake_chunks = [
        {
            "chunk_id": "paper__sec1__chunk0",
            "text":     "[Abstract]\nThe Transformer uses attention mechanisms entirely.",
            "metadata": {
                "citation": "Attention Is All You Need › Abstract (p1-1)",
                "filename": "sample_paper.pdf",
            },
            "score": 0.85,
        },
        {
            "chunk_id": "paper__sec4__chunk0",
            "text":     "[Multi-Head Attention]\nMulti-head attention runs h parallel heads.",
            "metadata": {
                "citation": "Attention Is All You Need › Multi-Head Attention (p4-5)",
                "filename": "sample_paper.pdf",
            },
            "score": 0.72,
        },
    ]

    result = build_grounded_prompt(
        question="What is the main idea of the transformer?",
        retrieved_chunks=fake_chunks,
    )

    rprint("[bold]System prompt (first 200 chars):[/bold]")
    rprint(result.messages[0]["content"][:200] + "…\n")

    rprint("[bold]User message (first 300 chars):[/bold]")
    rprint(result.messages[-1]["content"][:300] + "…\n")

    rprint(f"[bold]Sources prepared:[/bold] {len(result.sources)}")
    for s in result.sources:
        rprint(f"  [{s['ref']}] {s['citation']}  (score={s['score']})")
