"""
Streamlit UI — Agentic RAG Chatbot

Sidebar : file upload, document library, model + settings
Main    : chat interface with grounded answers and citation cards
"""

from __future__ import annotations
import re
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ingestion.parser          import parse_pdf
from app.ingestion.chunker         import chunk_document
from app.ingestion.embedder        import EmbedderClient
from app.generation.llm_client     import OllamaClient
from app.generation.prompt_builder import build_grounded_prompt, extract_cited_sources
from app.memory.memory_writer      import write_memory
from app.retrieval.hybrid          import hybrid_search


# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PaperMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ──────────────────────────────────────────────────────────
def _init():
    defaults = {
        "messages":     [],   # {role, content, sources}
        "last_sources": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── Cached resources (loaded once per session) ─────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder() -> EmbedderClient:
    return EmbedderClient(db_path="./data/chromadb")

@st.cache_resource(show_spinner=False)
def get_llm() -> OllamaClient:
    return OllamaClient()

embedder = get_embedder()
llm      = get_llm()


# ── Citation cards renderer ────────────────────────────────────────────────
def render_citations(sources: list[dict]):
    if not sources:
        return
    st.markdown("**Sources cited:**")
    cols = st.columns(min(len(sources), 3))
    for i, src in enumerate(sources):
        with cols[i % 3]:
            label = f"[{src['ref']}] {src['citation'][:45]}"
            with st.expander(label, expanded=False):
                st.caption(f"Relevance: {src['score']:.2f}")
                preview = src["text"][:350]
                if len(src["text"]) > 350:
                    preview += "…"
                # Strip section prefix for cleaner display
                preview = re.sub(r"^\[[^\]]+\]\n", "", preview)
                st.markdown(preview)


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🧠 PaperMind")
    st.caption("Ask your research. Get cited answers.")
    st.divider()

    # ── Ollama status ──────────────────────────────────────────────────────
    st.subheader("🤖 Model")
    if llm.is_available():
        models = llm.list_models()
        if models:
            chosen    = st.selectbox("Local model", models, index=0,
                                     label_visibility="collapsed")
            llm.model = chosen
            st.success(f"✓ Ollama running  |  {chosen}")
        else:
            st.warning("Ollama running but no models pulled.\n"
                       "Run: `ollama pull llama3.2`")
    else:
        st.error("Ollama not running.\n\nStart it:\n```\nollama serve\n```")

    st.divider()

    # ── File upload ────────────────────────────────────────────────────────
    st.subheader("📄 Upload PDF")
    uploaded = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded:
        if st.button("⚙️ Ingest Document", use_container_width=True,
                     type="primary"):
            with st.status(f"Ingesting {uploaded.name}…",
                           expanded=True) as status:
                try:
                    suffix = Path(uploaded.name).suffix
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix
                    ) as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name

                    st.write("📖 Parsing PDF…")
                    parsed = parse_pdf(tmp_path)
                    st.write(f"✓ {len(parsed.sections)} sections, "
                             f"{parsed.num_pages} pages")

                    st.write("✂️ Chunking…")
                    chunks = chunk_document(parsed)
                    st.write(f"✓ {len(chunks)} chunks")

                    st.write("🔢 Embedding…")
                    added = embedder.add_chunks(chunks, show_progress=False)
                    st.write(f"✓ {added} new chunks indexed")

                    write_memory(
                        user_questions=[],
                        indexed_docs=embedder.list_documents(),
                    )

                    status.update(
                        label=f"✅ Ingested: {uploaded.name}",
                        state="complete",
                    )
                    st.success(
                        f"**{parsed.title[:55]}**\n\n"
                        f"{len(parsed.sections)} sections · "
                        f"{added} chunks indexed"
                    )

                except Exception as e:
                    status.update(label="❌ Ingestion failed", state="error")
                    st.error(str(e))

    st.divider()

    # ── Document library ───────────────────────────────────────────────────
    st.subheader("🗂️ Document Library")
    docs = embedder.list_documents()

    if not docs:
        st.caption("No documents indexed yet.")
    else:
        for d in docs:
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(
                    f"**{d['doc_title'][:38]}**  \n"
                    f"<small>{d['chunks']} chunks · {d['filename']}</small>",
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("🗑", key=f"del_{d['doc_id']}",
                             help="Remove from index"):
                    embedder.delete_document(d["doc_id"])
                    st.rerun()

    st.caption(f"Total chunks in index: **{embedder.count()}**")
    st.divider()

    # ── Document scope selector ────────────────────────────────────────────
    # Dynamically built from the index — no hardcoding, works for any doc
    st.subheader("🔎 Search Scope")
    scope_options = ["All documents"] + [
        f"{d['doc_title'][:35]} ({d['doc_id']})"
        for d in embedder.list_documents()
    ]
    selected_scope = st.selectbox(
        "Search within",
        scope_options,
        index=0,
        help="Pin your question to one document, or search all.",
        label_visibility="collapsed",
    )
    active_doc_filter = (
        None if selected_scope == "All documents"
        else {"doc_id": selected_scope.split("(")[-1].rstrip(")")}
    )

    st.divider()

    # ── Settings ───────────────────────────────────────────────────────────
    st.subheader("⚙️ Settings")
    n_results   = st.slider("Chunks to retrieve", 3, 10, 8)
    min_score   = st.slider("Min relevance score", 0.0, 0.8, 0.3, 0.05)
    show_chunks = st.checkbox("Show raw retrieved chunks", value=False)

    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════════════════
st.header("💬 Ask PaperMind")

if not embedder.count():
    st.info("👆 Upload a PDF in the sidebar to get started.")

# ── Render chat history ────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_citations(msg["sources"])

# ── Chat input ─────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask PaperMind a question about your documents…"):

    if not embedder.count():
        st.warning("Please upload and ingest a document first.")
        st.stop()

    if not llm.is_available():
        st.error("Ollama is not running. Start it with `ollama serve`.")
        st.stop()

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ── Retrieve ─────────────────────────────────────────────────────────
    with st.spinner("🔍 Searching documents…"):
        retrieved = hybrid_search(
            query=prompt,
            client=embedder,
            n_results=n_results,
            where=active_doc_filter,
        )

    if show_chunks:
        with st.expander(
            f"📦 {len(retrieved)} chunks retrieved (pre-filter)",
            expanded=False,
        ):
            for c in retrieved:
                st.markdown(
                    f"**{c['metadata'].get('citation', '')}**  "
                    f"score `{c['score']}`"
                )
                st.caption(c["text"][:200])

    # ── Build prompt ──────────────────────────────────────────────────────
    # Each question is treated independently — no history injection.
    # This prevents cross-question context bleed, which is the correct
    # behavior for a document Q&A system where each query is self-contained.
    # RRF scores (0.01-0.04) are on a different scale than cosine scores (0-1).
    # Skip threshold filtering for hybrid results — retrieval ranking is enough.
    prompt_result = build_grounded_prompt(
        question=prompt,
        retrieved_chunks=retrieved,
        conversation_history=None,
        min_score_threshold=0.0,
    )

    # ── Generate ──────────────────────────────────────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("🤔 Generating answer…"):
            try:
                response = llm.chat(prompt_result.messages, temperature=0.1)
                answer   = response.text
            except RuntimeError as e:
                answer = f"⚠️ {e}"

        # Strip the "Sources: [1] ..." block the LLM appends to its answer.
        # We render citations ourselves via the citation cards below.
        answer_display = re.sub(
            r"\n+Sources:.*", "", answer, flags=re.DOTALL
        ).strip()

        # Suppress citation cards when the LLM issued a refusal
        _REFUSAL = "i could not find relevant information"
        is_refusal = _REFUSAL in answer.lower()
        cited      = [] if is_refusal else extract_cited_sources(
            answer, prompt_result.sources
        )

        st.markdown(answer_display)
        if cited:
            render_citations(cited)

    # ── Save to history ───────────────────────────────────────────────────
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer_display,   # store clean version
        "sources": cited,
    })

    # ── Update memory ─────────────────────────────────────────────────────
    all_questions = [
        m["content"]
        for m in st.session_state.messages
        if m["role"] == "user"
    ]
    write_memory(
        user_questions=all_questions,
        indexed_docs=embedder.list_documents(),
    )