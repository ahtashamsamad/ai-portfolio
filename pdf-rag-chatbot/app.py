"""PDF RAG Chatbot — ask questions about your PDFs, get grounded answers.

- Upload one or more PDFs (or try the bundled sample).
- Answers stream in live and cite page-numbered sources per document.
- The assistant only answers from the uploaded text: anything not found
  gets an honest "not in the document" reply instead of a hallucination.
- Optional OpenAI key (st.secrets first, sidebar fallback) upgrades
  extractive answers to natural-language ones. Without a key everything
  runs 100% locally — your documents never leave this machine.
"""

import io
import os

import streamlit as st

from rag_backend import (
    MAX_PDF_MB,
    RELEVANCE_THRESHOLD,
    chunk_pages,
    extract_pages,
    extractive_answer,
    llm_answer,
    not_in_document_message,
    stream_words,
    summarize,
    retrieve,
)

# ---------------------------------------------------------------------------
# Portfolio chrome
# ---------------------------------------------------------------------------

DEV_NAME = "Ahtasham Samad"
UPWORK_URL = "https://www.upwork.com/freelancers/~01d75561be7a2cd578"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PDF_PATH = os.path.join(BASE_DIR, "sample.pdf")


def get_secret_key():
    """Read OPENAI_API_KEY from st.secrets. Never hardcode keys in code."""
    try:
        return (st.secrets.get("OPENAI_API_KEY") or "").strip()
    except Exception:
        return ""


def render_about_sidebar():
    with st.sidebar:
        st.divider()
        st.subheader("About")
        st.markdown(f"**Built by [{DEV_NAME}]({UPWORK_URL})** — AI/ML Developer")
        st.markdown("**Tools:** Python, Streamlit, scikit-learn (TF-IDF "
                    "retrieval), pypdf, OpenAI API (optional)")


def render_footer():
    st.divider()
    st.markdown(
        f"Need a custom AI app? **[Hire me on Upwork]({UPWORK_URL})** · "
        f"Built by {DEV_NAME}"
    )


# ---------------------------------------------------------------------------
# Document handling
# ---------------------------------------------------------------------------

def validate_pdf(uploaded):
    """Return an error string for a bad file, or '' when it looks fine."""
    if not uploaded.name.lower().endswith(".pdf"):
        return f"'{uploaded.name}' is not a PDF file."
    size = getattr(uploaded, "size", 0) or 0
    if size == 0:
        return f"'{uploaded.name}' is empty (0 bytes)."
    if size > MAX_PDF_MB * 1024 * 1024:
        return (f"'{uploaded.name}' is {size / 1024 / 1024:.1f} MB — "
                f"the limit is {MAX_PDF_MB} MB per file.")
    return ""


def index_documents(files):
    """Extract + chunk every file. Returns (docs, problems).

    docs: {name: {"pages": int, "chunks": [...], "text": str}}
    problems: {name: friendly per-file error string}.
    """
    docs, problems = {}, {}
    with st.spinner(f"Reading {len(files)} PDF(s)…"):
        progress = st.progress(0.0)
        for i, f in enumerate(files):
            err = validate_pdf(f)
            if err:
                problems[f.name] = err
            else:
                try:
                    pages = extract_pages(f)
                except RuntimeError as exc:      # password-protected
                    problems[f.name] = f"{f.name}: {exc}"
                except Exception:
                    problems[f.name] = (f"{f.name}: could not be read — "
                                        "the file may be corrupt.")
                else:
                    if not pages:
                        problems[f.name] = (
                            f"{f.name}: no extractable text found. It may be a "
                            "scanned (image-only) PDF — run it through an OCR "
                            "tool first, then re-upload.")
                    else:
                        full = " ".join(t for _, t in pages)
                        docs[f.name] = {"pages": len(pages),
                                        "chunks": chunk_pages(pages),
                                        "text": full}
            progress.progress((i + 1) / len(files))
        progress.empty()
    return docs, problems


def sample_file():
    """Load the bundled sample PDF as an upload-like object."""
    with open(SAMPLE_PDF_PATH, "rb") as f:
        data = f.read()
    bio = io.BytesIO(data)
    bio.name = "sample.pdf"
    bio.size = len(data)
    return bio


def build_answer(question, docs, api_key):
    """Return (sources, answer_text) for a question over the indexed docs."""
    corpus = []
    for name, d in docs.items():
        for c in d["chunks"]:
            corpus.append({"doc": name, **c})
    results = retrieve(corpus, question)
    if not results or results[0]["score"] < RELEVANCE_THRESHOLD:
        return [], not_in_document_message()
    if api_key:
        try:
            return results, llm_answer(question, results, api_key)
        except Exception as exc:  # bad key, no network, quota…
            return results, (
                f"The AI service failed ({exc}). "
                "Showing the most relevant passages instead.\n\n"
                + extractive_answer(results))
    return results, extractive_answer(results)


def render_history():
    for m in st.session_state.messages:
        with st.chat_message("user" if m["role"] == "user" else "assistant"):
            st.markdown(m["content"])
            for i, r in enumerate(m.get("sources", []), 1):
                snippet = r["text"][:1200] + ("…" if len(r["text"]) > 1200 else "")
                with st.expander(
                    f"📄 Source {i} — {r['doc']}, page {r['page']} "
                    f"(relevance {r['score']:.3f})"
                ):
                    st.write(snippet)


def chat_as_text():
    lines = []
    for m in st.session_state.messages:
        who = "You" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m['content']}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "docs" not in st.session_state:
        st.session_state.docs = {}          # name -> indexed document
    if "indexed_names" not in st.session_state:
        st.session_state.indexed_names = ()


def main():
    st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄",
                       layout="wide")
    init_state()

    st.title("📄 PDF RAG Chatbot")
    st.caption("⏳ If the app was asleep, it may take up to a minute to wake — "
               "please wait.")
    st.info(
        "**What it does:** Chat with your PDFs — grounded answers with "
        "page-numbered sources, never hallucinations.\n\n"
        "**Who it's for:** Students, researchers and professionals working "
        "with long documents.\n\n"
        "**How to use:** ① Upload one or more PDFs (or try the sample) → "
        "② Ask a question → ③ Check the cited sources."
    )

    secret_key = get_secret_key()
    with st.sidebar:
        st.header("Settings")
        if secret_key:
            st.success("OpenAI key loaded from secrets ✓")
            api_key = secret_key
        else:
            api_key = st.text_input(
                "OpenAI API key (optional)", type="password",
                help="With a key, an LLM writes a natural answer from the "
                     "retrieved passages. Without one, the app shows the "
                     "passages directly.")
            api_key = (api_key or "").strip()
        st.caption("Extractive mode (no key) runs 100% locally — your "
                   "documents never leave this machine.")
    render_about_sidebar()

    # ---- inputs: uploads + sample ----
    col_up, col_sample = st.columns([3, 1])
    with col_up:
        uploads = st.file_uploader(
            "Upload PDF(s)", type=["pdf"], accept_multiple_files=True,
            help=f"PDF only, up to {MAX_PDF_MB} MB per file.")
    with col_sample:
        st.write("")  # align with uploader
        if st.button("✨ Try with sample",
                     help="Load a bundled sample PDF — no upload needed."):
            st.session_state.sample_requested = True

    files = list(uploads or [])
    if st.session_state.pop("sample_requested", False):
        files = [sample_file()]
        st.info("Sample PDF loaded — ask it anything!")

    names = tuple(sorted(f.name for f in files))
    if names != st.session_state.indexed_names:
        # Document set changed -> (re)index everything.
        st.session_state.messages = []
        if not files:
            st.session_state.docs = {}
            st.session_state.indexed_names = ()
            st.info("Upload a PDF (or try the sample) to get started.")
            render_footer()
            return
        docs, problems = index_documents(files)
        for problem in problems.values():
            st.error(problem)
        if not docs:
            st.session_state.docs = {}
            st.session_state.indexed_names = names
            render_footer()
            return
        st.session_state.docs = docs
        st.session_state.indexed_names = names
        total_pages = sum(d["pages"] for d in docs.values())
        total_chunks = sum(len(d["chunks"]) for d in docs.values())
        st.success(f"Indexed {total_chunks} passages from "
                   f"{len(docs)} document(s), {total_pages} pages.")

    docs = st.session_state.docs

    # ---- document summary ----
    with st.expander("📝 Document summary (auto-generated)", expanded=False):
        for name, d in docs.items():
            st.markdown(f"**{name}** — {d['pages']} pages, "
                        f"{len(d['chunks'])} passages")
            st.write(summarize(d["text"]) or "Summary unavailable.")

    # ---- chat ----
    head_l, head_r = st.columns([4, 1])
    with head_l:
        st.caption("Chatting with: " + ", ".join(
            f"**{n}**" for n in docs))
    with head_r:
        if st.button("🧹 Clear chat",
                     help="Clear the chat history (keeps the indexed PDFs)"):
            st.session_state.messages = []
            st.rerun()

    render_history()

    if prompt := st.chat_input("Ask a question about the document(s)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("Searching the document(s)…"):
            sources, answer = build_answer(prompt, docs, api_key)
        with st.chat_message("assistant"):
            st.write_stream(stream_words(answer))   # visible streaming
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources})
        st.rerun()

    if st.session_state.messages:
        st.download_button(
            "⬇ Download chat as TXT", chat_as_text().encode("utf-8"),
            file_name="pdf-chat.txt", mime="text/plain")

    render_footer()


def _in_streamlit_runtime():
    try:
        from streamlit.runtime import exists
        return bool(exists())
    except Exception:
        return False


if _in_streamlit_runtime():
    main()
