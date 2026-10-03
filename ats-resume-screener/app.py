"""ATS Resume Screener — Streamlit app.

Ranks resumes against a job description using TF-IDF semantic similarity
plus skill overlap, and explains every score. Deployable on Streamlit
Community Cloud: main file path is app.py.
"""

import glob
import os

import pandas as pd
import streamlit as st
from pypdf import PdfReader

from screener import load_skills, name_from_filename, rank_resumes

BASE_DIR = os.path.dirname(__file__)
SAMPLE_JD_PATH = os.path.join(BASE_DIR, "sample_jd.txt")
SAMPLE_RESUMES_DIR = os.path.join(BASE_DIR, "resumes")

st.set_page_config(page_title="ATS Resume Screener", page_icon="📋", layout="wide")

st.title("📋 ATS Resume Screener")
st.write(
    "HR teams receive hundreds of resumes per posting, and keyword-only ATS "
    "filters reject strong candidates over vocabulary mismatch. Paste a job "
    "description, upload resumes, and get a **ranked shortlist with every "
    "score explained** — TF-IDF semantic similarity plus skill overlap. "
    "No black box."
)

SKILLS = load_skills()

if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""
if "resume_files" not in st.session_state:
    st.session_state.resume_files = []  # list of (name, text)


def read_text_file(uploaded):
    """Extract text from an uploaded .txt or .pdf file."""
    name = uploaded.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(uploaded)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return uploaded.read().decode("utf-8", errors="ignore")


def load_sample_data():
    with open(SAMPLE_JD_PATH, encoding="utf-8") as f:
        jd = f.read()
    resumes = []
    for path in sorted(glob.glob(os.path.join(SAMPLE_RESUMES_DIR, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            resumes.append((name_from_filename(path), f.read()))
    return jd, resumes


# ---------- sample data ----------
if st.button("✨ Load sample data (AI/ML Engineer posting + 5 resumes)"):
    jd, resumes = load_sample_data()
    st.session_state.jd_text = jd
    st.session_state.resume_files = resumes
    st.session_state.pop("ranked", None)
    st.success(f"Loaded job description + {len(resumes)} sample resumes.")

# ---------- inputs ----------
col_jd, col_res = st.columns(2)

with col_jd:
    st.subheader("1️⃣ Job description")
    jd_text = st.text_area(
        "Paste the job description", height=280, key="jd_text",
        help="Or upload it below — the upload replaces this text.",
    )
    jd_upload = st.file_uploader("…or upload JD (.txt / .pdf)", type=["txt", "pdf"], key="jd_up")
    if jd_upload is not None:
        jd_text = read_text_file(jd_upload)

with col_res:
    st.subheader("2️⃣ Resumes")
    uploads = st.file_uploader(
        "Upload resumes (.pdf / .txt)", type=["pdf", "txt"],
        accept_multiple_files=True, key="res_up",
    )
    if uploads:
        st.session_state.resume_files = [
            (name_from_filename(u.name), read_text_file(u)) for u in uploads
        ]
        st.session_state.pop("ranked", None)
    if st.session_state.resume_files:
        names = [n for n, _ in st.session_state.resume_files]
        st.write(f"**{len(names)}** resume(s) ready: " + ", ".join(names))
        empty = [n for n, t in st.session_state.resume_files if not t.strip()]
        if empty:
            st.warning(
                "No extractable text in: " + ", ".join(empty)
                + " (scanned/image PDFs need OCR first)."
            )

# ---------- ranking ----------
st.subheader("3️⃣ Rank candidates")
if st.button("🚀 Rank candidates", type="primary"):
    if not jd_text.strip():
        st.warning("Paste a job description (or load the sample data) first.")
    elif not st.session_state.resume_files:
        st.warning("Upload at least one resume (or load the sample data).")
    else:
        with st.spinner("Scoring candidates…"):
            ranked = rank_resumes(
                jd_text,
                [{"name": n, "text": t} for n, t in st.session_state.resume_files],
                skills=SKILLS,
            )
        st.session_state.ranked = ranked

if st.session_state.get("ranked"):
    ranked = st.session_state.ranked
    st.subheader("📊 Ranked shortlist")
    df = pd.DataFrame(
        [
            {
                "Rank": i + 1,
                "Candidate": r["name"],
                "Match score (%)": r["score"],
                "Matched skills": len(r["matched_skills"]),
                "Missing skills": len(r["missing_skills"]),
                "Text similarity": r["tfidf_similarity"],
            }
            for i, r in enumerate(ranked)
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    jd_skills = ranked[0]["jd_skills"]
    with st.expander(f"🔎 Skills detected in the job description ({len(jd_skills)})"):
        st.write(", ".join(jd_skills) if jd_skills else "None detected.")

    for i, r in enumerate(ranked, start=1):
        with st.expander(f"#{i} — {r['name']} · {r['score']}% match"):
            st.progress(min(1.0, r["score"] / 100))
            c1, c2 = st.columns(2)
            with c1:
                st.metric("TF-IDF similarity", f"{r['tfidf_similarity']:.3f}")
            with c2:
                st.metric("Skill overlap", f"{r['skill_overlap']:.0%}")
            st.markdown("**✅ Matched skills**")
            st.write(", ".join(r["matched_skills"]) if r["matched_skills"] else "—")
            st.markdown("**❌ Missing skills (wanted by the JD)**")
            st.write(", ".join(r["missing_skills"]) if r["missing_skills"] else "—")
            st.info(r["summary"])

    st.caption(
        "Scores rank candidates for **shortlisting only** — they are similarity "
        "measurements, not hiring decisions. Always have a human review the shortlist."
    )

with st.sidebar:
    st.header("How scoring works")
    st.write(
        "Score = **0.6 × TF-IDF cosine similarity** (job description vs resume "
        "wording) + **0.4 × skill overlap** (share of the JD's listed skills "
        "found in the resume), scaled to 0–100.\n\n"
        "Common abbreviations are expanded before matching (ML → machine "
        "learning), so candidates aren't punished for vocabulary mismatch — "
        "the exact problem keyword-only ATS systems have."
    )
