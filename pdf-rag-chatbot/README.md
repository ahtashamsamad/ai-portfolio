Live demo: https://ai-portfolio-jwev4baznf4cjwf5hv6wdx.streamlit.app/

# 📄 PDF RAG Chatbot

Upload any PDF and ask questions in plain English. The app retrieves the most
relevant passages and **shows the sources** (page number + relevance score) so
every answer can be verified. Add an OpenAI API key in the sidebar for
natural-language answers written only from those passages.

## Features

- Modern chat UI (`st.chat_message` / `st.chat_input`) with the full
  conversation kept in `st.session_state` — the PDF is indexed **once**,
  then reused on every rerun
- PDF upload with page-by-page text extraction (`pypdf`)
- **Sentence-aware chunking**: ~300-word passages that break at sentence
  boundaries (50-word overlap), each tagged with its page number
- TF-IDF retrieval (`scikit-learn`) — top-3 passages per question
- **Extractive mode** (no key): shows the passages directly, 100% local —
  your document never leaves the machine
- **LLM mode** (optional key): GPT writes a natural answer grounded *only*
  in the retrieved passages, with page citations; graceful fallback to
  extractive mode if the API call fails
- Upload guardrails: 25 MB file-size limit, plus friendly errors (not
  crashes) for corrupt, password-protected, or scanned/image-only PDFs
- 🧹 **Clear chat** button — resets the conversation while the PDF stays indexed

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## How it works

1. **Split** — the PDF is extracted page by page and split into overlapping
   ~300-word passages, each remembering its page number.
2. **Retrieve** — your question is vectorized with TF-IDF and matched
   against all passages by cosine similarity; the top 3 win.
3. **Answer** — in extractive mode you read the winning passages yourself;
   in LLM mode GPT-4o-mini rewrites them into a natural answer, citing
   pages like [page 2], and says so honestly when the answer isn't there.

## Deploying (Streamlit Community Cloud)

Push this folder to GitHub, then at share.streamlit.io → New app → pick the
repo → set **Main file path** to `pdf-rag-chatbot/app.py` → Deploy.

## Project structure

```
pdf-rag-chatbot/
├── app.py            # Streamlit UI (chat, session state)
├── rag_backend.py    # extraction, chunking, retrieval, LLM (testable headless)
├── requirements.txt
└── README.md
```
