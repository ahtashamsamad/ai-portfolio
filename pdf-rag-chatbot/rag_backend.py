"""PDF RAG backend — pure functions, no Streamlit dependency.

Importable and testable headlessly. The Streamlit UI lives in app.py.
"""


import re

from pypdf import PdfReader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNK_WORDS = 300      # target words per passage
OVERLAP_WORDS = 50     # overlap between consecutive passages (keeps context)
TOP_K = 3              # how many passages to retrieve per question
MAX_PDF_MB = 25        # reject uploads larger than this (memory safety)
RELEVANCE_THRESHOLD = 0.25  # top passage below this -> "not in the document"

# Split on sentence-ending punctuation so chunks break at sentence
# boundaries instead of mid-sentence.
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Core logic (pure functions — easy to test, no Streamlit dependency)
# ---------------------------------------------------------------------------

def extract_pages(pdf_file):
    """Return [(page_number, text), ...] for every page with extractable text.

    `pdf_file` is any binary file-like object (e.g. Streamlit's uploader).
    Pages with no text (scanned images) are skipped.
    Raises RuntimeError for password-protected PDFs.
    """
    reader = PdfReader(pdf_file)
    if reader.is_encrypted:
        raise RuntimeError("This PDF is password-protected. Remove the "
                           "password first, then re-upload.")
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_pages(pages, chunk_words=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    """Split page texts into sentence-aware overlapping chunks.

    Sentences are packed greedily up to ~chunk_words; consecutive chunks
    share the last ~overlap words so an idea is never cut off mid-thought.
    A single pathological sentence longer than chunk_words is word-sliced.
    Returns [{"page": int, "text": str}, ...].
    """
    chunks = []

    def flush(buf, page_num):
        if buf:
            chunks.append({"page": page_num, "text": " ".join(buf)})

    for page_num, text in pages:
        sentences = [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]
        if not sentences:
            continue
        buf, buf_words = [], 0
        for sent in sentences:
            words = sent.split()
            if len(words) > chunk_words:
                # Pathological single sentence: fall back to word slicing.
                flush(buf, page_num)
                buf, buf_words = [], 0
                step = max(1, chunk_words - overlap)
                for i in range(0, len(words), step):
                    flush(words[i:i + chunk_words], page_num)
                continue
            if buf_words + len(words) > chunk_words:
                flush(buf, page_num)
                # Carry trailing overlap words into the next chunk so
                # context survives the boundary.
                buf = buf[-overlap:] if overlap > 0 else []
                buf_words = len(buf)
            buf.extend(words)
            buf_words += len(words)
        flush(buf, page_num)
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


def extractive_answer(results):
    """Compose a plain-text answer from retrieved passages (no LLM needed)."""
    lines = ["Based on the uploaded document(s):", ""]
    for i, r in enumerate(results, 1):
        snippet = re.sub(r"\s+", " ", r["text"]).strip()
        if len(snippet) > 500:
            snippet = snippet[:500] + "…"
        lines.append(f"[{i}] {r['doc']} — page {r['page']}: {snippet}")
        lines.append("")
    lines.append("Open a source below to read the full passage.")
    return "\n".join(lines)


def stream_words(text, chunk_size=8):
    """Yield text in small word chunks for st.write_stream (visible streaming)."""
    words = text.split(" ")
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size]) + " "


def summarize(text, n_sentences=5):
    """Extractive summary: top-N sentences by word-frequency scoring.

    Pure function — `text` is the document's full plain text.
    """
    from collections import Counter

    sentences = [s.strip() for s in SENTENCE_RE.split(text)
                 if len(s.split()) > 4]
    if not sentences:
        return ""
    words = re.findall(r"[a-z]+", text.lower())
    freq = Counter(w for w in words if w not in ENGLISH_STOP_WORDS)
    if not freq:
        return " ".join(sentences[:n_sentences])

    def score(sentence):
        ws = [w for w in re.findall(r"[a-z]+", sentence.lower())
              if w not in ENGLISH_STOP_WORDS]
        return sum(freq[w] for w in ws) / max(1, len(ws))

    top = sorted(sentences, key=score, reverse=True)[:n_sentences]
    order = {s: i for i, s in enumerate(sentences)}
    return " ".join(sorted(top, key=lambda s: order[s]))


def not_in_document_message():
    """Honest reply when nothing relevant was retrieved."""
    return ("I couldn't find an answer to that in the uploaded document(s). "
            "I only answer from the document text — try rephrasing your "
            "question, or ask about something the document covers.")


# ---------------------------------------------------------------------------
