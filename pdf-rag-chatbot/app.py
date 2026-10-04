"""
PDF RAG Chatbot — ask questions about any PDF, get answers with sources.

How it works:
  1. Upload a PDF -> text is extracted page by page (pypdf).
  2. Text is split into ~300-word passages (50-word overlap), each tagged
     with the page number it came from.
  3. Your question is matched against the passages with TF-IDF
     (scikit-learn); the top-3 passages are shown with page number and
     a relevance score.
  4. Optionally, paste an OpenAI API key in the sidebar and an LLM will
     write a natural-language answer grounded ONLY in those passages.
     Without a key the app runs in extractive mode: it just shows you
     the most relevant passages directly (nothing leaves your machine).
  5. The indexed document and the full chat history live in
     st.session_state — the PDF is processed once and the conversation
     survives every rerun.
"""

import streamlit as st

from rag_backend import (
    MAX_PDF_MB,
    chunk_pages,
    extract_pages,
    llm_answer,
    retrieve,
)


# Session state — chat history + indexed document survive every rerun
# ---------------------------------------------------------------------------

def init_state():
    """Create the session-state keys this app depends on (idempotent)."""
    if "messages" not in st.session_state:
        # [{"role": "user"|"assistant", "content": str, "sources": [...]}, ...]
        st.session_state.messages = []
    if "doc" not in st.session_state:
        # {"name": str, "pages": int, "chunks": [...]} once a PDF is indexed
        st.session_state.doc = None


def index_pdf(uploaded):
    """Extract + chunk a newly uploaded PDF and store it in session state.

    Returns True on success, False when the file is rejected (too large,
    corrupt, encrypted, or no extractable text). All failure modes show a
    friendly message instead of crashing.
    """
    size = getattr(uploaded, "size", 0) or 0
    if size > MAX_PDF_MB * 1024 * 1024:
        st.error(f"That PDF is {size / 1024 / 1024:.1f} MB — the limit is "
                 f"{MAX_PDF_MB} MB. Please upload a smaller file.")
        return False
    if size == 0:
        st.error("That file looks empty (0 bytes). Please upload a valid PDF.")
        return False
    try:
        with st.spinner("Reading PDF…"):
            pages = extract_pages(uploaded)
    except RuntimeError as exc:
        # password-protected PDF (raised by extract_pages)
        st.error(str(exc))
        return False
    except Exception:
        st.error("Could not read this PDF — the file may be corrupt. "
                 "Try re-exporting it, then re-upload.")
        return False
    if not pages:
        st.warning("No extractable text found in this PDF. It may be a scanned "
                   "document (images only) — run it through an OCR tool first, "
                   "then re-upload.")
        return False
    chunks = chunk_pages(pages)
    st.session_state.doc = {
        "name": uploaded.name,
        "pages": len(pages),
        "chunks": chunks,
    }
    st.session_state.messages = []       # fresh document -> fresh conversation
    st.success(f"Indexed {len(chunks)} passages from {len(pages)} pages.")
    return True


def answer_question(question, api_key):
    """Run retrieval (+ optional LLM) and append the exchange to the history."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.spinner("Searching…"):
        results = retrieve(st.session_state.doc["chunks"], question)

    sources = results
    if not results or all(r["score"] <= 0 for r in results):
        content = ("No relevant passages found in this document. "
                   "Try rephrasing your question.")
        sources = []
    elif api_key:
        try:
            with st.spinner("Writing answer…"):
                content = llm_answer(question, results, api_key)
        except Exception as exc:  # e.g. bad key, no network
            content = (f"OpenAI call failed ({exc}). "
                       f"Showing the retrieved passages instead.")
    else:
        content = ("Extractive mode — top matching passages below "
                   "(add an OpenAI key in the sidebar for natural-language answers).")

    st.session_state.messages.append(
        {"role": "assistant", "content": content, "sources": sources}
    )


def render_history():
    """Render the conversation stored in session state as chat bubbles."""
    for m in st.session_state.messages:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.markdown(m["content"])
            for i, r in enumerate(m.get("sources", []), 1):
                snippet = r["text"][:1200] + ("…" if len(r["text"]) > 1200 else "")
                with st.expander(
                    f"📄 Passage {i} — page {r['page']} (relevance {r['score']:.3f})"
                ):
                    st.write(snippet)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄")
    st.title("📄 Chat with your PDF")
    st.caption("Upload a PDF, ask questions in plain English, and see exactly "
               "which passages the answers come from.")

    init_state()

    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input(
            "OpenAI API key (optional)",
            type="password",
            help="With a key, an LLM writes a natural answer from the retrieved "
                 "passages. Without one, the app shows the passages directly.",
        )
        st.caption("Extractive mode (no key) runs 100% locally — your document "
                   "never leaves this machine.")

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded is None:
        # File removed (or never uploaded) -> drop the document and history.
        st.session_state.doc = None
        st.session_state.messages = []
        st.info("Upload a PDF to get started.")
        return

    if st.session_state.doc is None or st.session_state.doc["name"] != uploaded.name:
        # New (or first) document — index once, then reuse from session state.
        if not index_pdf(uploaded):
            return

    doc = st.session_state.doc
    head_l, head_r = st.columns([5, 1])
    with head_l:
        st.caption(f"Chatting with **{doc['name']}** — {doc['pages']} pages indexed.")
    with head_r:
        if st.button("🧹 Clear", help="Clear the chat history (keeps the indexed PDF)"):
            st.session_state.messages = []
            st.rerun()

    render_history()

    # st.chat_input clears itself after submit, so each question is
    # processed exactly once — no rerun guard needed.
    if prompt := st.chat_input("Ask a question about the document"):
        answer_question(prompt, api_key)
        st.rerun()


def _in_streamlit_runtime():
    """True only when executed via `streamlit run` (lets us import headlessly)."""
    try:
        from streamlit.runtime import exists
        return bool(exists())
    except Exception:
        return False


if _in_streamlit_runtime():
    main()
