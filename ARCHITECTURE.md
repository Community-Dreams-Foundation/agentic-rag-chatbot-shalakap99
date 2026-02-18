# Architecture Overview

## High-Level Flow
````
PDF Upload
    │
    ▼
┌─────────────────────────────────────────────┐
│  Ingestion Pipeline                          │
│  parse_pdf() → chunk_document() →           │
│  EmbedderClient.add_chunks()               │
│                                             │
│  • Section-aware parsing (PyMuPDF)         │
│  • Sentence-boundary chunking + overlap    │
│  • all-MiniLM-L6-v2 embeddings (384-dim)  │
│  • ChromaDB persistent local storage       │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Retrieval                                   │
│  Dense vector search (cosine similarity)    │
│  Metadata filters (doc_id, section)        │
└──────────────────┬──────────────────────────┘
                   │  top-k chunks + scores
                   ▼
┌─────────────────────────────────────────────┐
│  Grounded Generation                         │
│  build_grounded_prompt() →                  │
│  OllamaClient (llama3.2, local) →          │
│  extract_cited_sources()                   │
└──────────────────┬──────────────────────────┘
                   │  answer + [1][2] citations
                   ▼
┌─────────────────────────────────────────────┐
│  Memory Writer                               │
│  USER_MEMORY.md  — questions + topics       │
│  COMPANY_MEMORY.md — KB state + doc list   │
└─────────────────────────────────────────────┘
````

---

## 1. Ingestion (`app/ingestion/`)

### Parsing — `parser.py`
- **Library:** PyMuPDF (`fitz`) for layout-preserving text extraction
- **Heading detection:** two-signal heuristic
  - Font size ≥ 1.25× body font size (estimated from most common size across first 5 pages)
  - OR matches a set of known arXiv section names (`Abstract`, `Introduction`, `Methods`, etc.)
  - Page 1 excluded from font-size detection to avoid author affiliation lines
- **Noise filtering:** regex blocklist removes arXiv IDs, formula fragments, bare page numbers
- **Post-processing:** sections with fewer than 30 words are merged into the previous section
- **Fallback:** page-level splitting if fewer than 3 sections detected
- **Metadata per section:** `title`, `page_start`, `page_end`, `section_index`

### Chunking — `chunker.py`
- **Primary:** one chunk per section if it fits within `max_chars=1200`
- **Overflow:** sentence-boundary splitting with `overlap_chars=150` carried into the next chunk
- **Section prefix:** every chunk is prefixed with `[SectionName]` so context survives retrieval
- **Citation property:** each `Chunk` has a `.citation` string: `"Title › Section (pN–M)"`
- **ChromaDB metadata:** flat dict with `doc_id`, `section_title`, `page_start`, `page_end`, `citation`

### Embedding — `embedder.py`
- **Model:** `all-MiniLM-L6-v2` — 22M params, 384-dim, runs on CPU, ~80MB
- **Vector store:** ChromaDB with HNSW index, cosine similarity, persistent to `data/chromadb/`
- **Normalization:** `normalize_embeddings=True` so dot product = cosine similarity
- **Idempotency:** chunks are keyed by `chunk_id`; re-ingesting is safe and skips duplicates
- **Score conversion:** ChromaDB cosine distance (0–2) → similarity score (0–1)

---

## 2. Retrieval

**Current:** dense vector search via ChromaDB
- Query embedded with same model as documents
- Top-k chunks returned by cosine similarity
- Chunks below `min_score_threshold` (default 0.30) discarded before LLM prompt

**Metadata filters:** `where={"doc_id": "paper_abc"}` scopes search to one document

**Planned (next iteration):**
- BM25 lexical search via `rank-bm25`
- Reciprocal Rank Fusion (RRF) to merge dense + lexical ranked lists
- Cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) for final top-5 selection

---

## 3. Grounded Generation + Citations

### Prompt design — `prompt_builder.py`
The system prompt enforces three hard rules:
1. Answer ONLY from provided context — no outside knowledge
2. Cite every claim inline: `[1]`, `[2]`, `[3]`
3. If answer not in context → respond with exact refusal string (no hallucination path)

Each retrieved chunk is numbered in the context block so `[N]` references are unambiguous.

### Citation extraction
`extract_cited_sources()` parses `[N]` patterns from the LLM response and returns only the chunks that were actually cited. This powers the expandable citation cards in the UI.

### Citation schema (matches judge validator)
````json
{ "source": "filename.pdf", "locator": "Section › Title (pN–M)", "snippet": "..." }
````

### LLM — `llm_client.py`
- Targets locally running Ollama (`http://localhost:11434`)
- Default model: `llama3.2` (3.2B, Q4_K_M, runs on CPU)
- Temperature: `0.1` — near-deterministic for factual grounded answers
- Graceful failure: `make sanity` works without Ollama via retrieval-only fallback

---

## 4. Memory System (`app/memory/memory_writer.py`)

### USER_MEMORY.md
- Written after every answered question
- Tracks: questions asked, inferred topics (keyword frequency, stopword-filtered)
- Does NOT store: PII, raw conversation transcripts, API keys

### COMPANY_MEMORY.md
- Written after every ingestion event
- Tracks: indexed documents, chunk counts, embedding model, vector store type
- Reusable across users of the same deployment

### What counts as high-signal memory
- Questions asked → reveals user research focus
- Documents ingested → reveals knowledge base composition
- Explicitly NOT stored: intermediate retrieval results, low-confidence inferences

---

## 5. Tradeoffs & What I'd Improve Next

| Decision | Tradeoff |
|---|---|
| `all-MiniLM-L6-v2` embeddings | Fast + free + CPU; lower quality than `text-embedding-3-small` |
| Section-aware chunking | Better citation granularity; slower to parse than fixed-size |
| ChromaDB local | Zero infrastructure; not horizontally scalable |
| Ollama local LLM | No API cost; first inference slow on CPU |
| Dense-only retrieval | Simple to ship; misses exact keyword matches that BM25 catches |
| Retrieval-only sanity fallback | Always passes `make sanity`; answer quality lower without LLM |

### With more time
- Hybrid BM25 + dense retrieval with RRF fusion
- Cross-encoder reranker for top-5 selection
- Streaming responses in Streamlit UI
- Confidence-gated memory writes (LLM self-evaluates before writing)
- Evaluation harness against `EVAL_QUESTIONS.md` with assertion-based citation checks
- Knowledge-graph overlay for multi-hop questions
