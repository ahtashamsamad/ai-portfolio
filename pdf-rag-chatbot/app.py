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
"""

import re

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNK_WORDS = 300      # target words per passage
OVERLAP_WORDS = 50     # overlap between consecutive passages (keeps context)
TOP_K = 3              # how many passages to retrieve per question


# ---------------------------------------------------------------------------
# Core logic (pure functions — easy to test, no Streamlit dependency)
# ---------------------------------------------------------------------------

def extract_pages(pdf_file):
    """Return [(page_number, text), ...] for every page with extractable text.

    `pdf_file` is any binary file-like object (e.g. Streamlit's uploader).
    Pages with no text (scanned images) are skipped.
    """
    reader = PdfReader(pdf_file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_pages(pages, chunk_words=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    """Split page texts into overlapping word chunks, keeping page numbers.

    Returns [{"page": int, "text": str}, ...].
    """
    chunks = []
    for page_num, text in pages:
        words = text.split()
        if not words:
            continue
        start = 0
        while start < len(words):
            end = start + chunk_words
            chunks.append({"page": page_num, "text": " ".join(words[start:end])})
            if end >= len(words):
                break
            start = end - overlap
    return chunks


def retrieve(chunks, question, top_k=TOP_K):
    """Rank chunks against the question using TF-IDF cosine similarity.

    Returns [{"page", "text", "score"}] sorted by descending relevance.
    """
    if not chunks or not question.strip():
        return []
    corpus = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_matrix = vectorizer.fit_transform(corpus)
    query_vec = vectorizer.transform([question])
    scores = cosine_similarity(query_vec, doc_matrix)[0]
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [
        {"page": chunks[idx]["page"], "text": chunks[idx]["text"], "score": float(score)}
        for idx, score in ranked[:top_k]
    ]


def llm_answer(question, passages, api_key):
    """Write a natural answer with OpenAI, grounded ONLY in the passages.

    The import is lazy so the OpenAI package is never touched unless a key
    is actually provided.
    """
    from openai import OpenAI  # imported here: only needed with a key

    context = "\n\n".join(
        f"[Passage {i + 1} — page {p['page']}]\n{p['text']}"
        for i, p in enumerate(passages)
    )
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions using ONLY the passages below. "
                    "If the passages do not contain the answer, say so honestly. "
                    "Cite the page number of each claim like [page 2]."
                ),
            },
            {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄")
    st.title("📄 Chat with your PDF")
    st.caption("Upload a PDF, ask questions in plain English, and see exactly "
               "which passages the answers come from.")

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
        st.info("Upload a PDF to get started.")
        return

    with st.spinner("Reading PDF…"):
        pages = extract_pages(uploaded)

    if not pages:
        st.warning("No extractable text found in this PDF. It may be a scanned "
                   "document (images only) — run it through an OCR tool first, "
                   "then re-upload.")
        return

    chunks = chunk_pages(pages)
    st.success(f"Indexed {len(chunks)} passages from {len(pages)} pages.")

    question = st.text_input("Ask a question about the document")
    if not question:
        return

    with st.spinner("Searching…"):
        results = retrieve(chunks, question)

    if not results or all(r["score"] <= 0 for r in results):
        st.warning("No relevant passages found. Try rephrasing your question.")
        return

    if api_key:
        try:
            with st.spinner("Writing answer…"):
                answer = llm_answer(question, results, api_key)
            st.subheader("Answer")
            st.write(answer)
        except Exception as exc:  # e.g. bad key, no network
            st.error(f"OpenAI call failed ({exc}). Showing retrieved passages instead.")
    else:
        st.info("Extractive mode — top matching passages (add an OpenAI key in "
                "the sidebar for natural-language answers).")

    st.subheader("Sources")
    for i, r in enumerate(results, 1):
        snippet = r["text"][:1200] + ("…" if len(r["text"]) > 1200 else "")
        with st.expander(f"Passage {i} — page {r['page']} (relevance {r['score']:.3f})"):
            st.write(snippet)


def _in_streamlit_runtime():
    """True only when executed via `streamlit run` (lets us import headlessly)."""
    try:
        from streamlit.runtime import exists
        return bool(exists())
    except Exception:
        return False


if _in_streamlit_runtime():
    main()
