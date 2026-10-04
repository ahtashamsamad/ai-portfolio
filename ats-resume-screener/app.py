"""ATS Resume Screener — Streamlit app.

Ranks resumes against a job description using TF-IDF semantic similarity
plus skill overlap, and explains every score: gauge chart, matched/missing
skill pills, section-wise feedback and concrete improvement suggestions.
Recruiters can screen many resumes at once; candidates get actionable
feedback. Deployable on Streamlit Community Cloud: main file path is app.py.

Privacy: resumes are processed in memory only — never stored, never sent
anywhere.
"""

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from screener import (
    analyze_sections,
    build_report_pdf,
    build_suggestions,
    load_sample_data,
    load_skills,
    name_from_filename,
    rank_resumes,
    read_text_file,
)

# ---------------------------------------------------------------------------
# Portfolio chrome
# ---------------------------------------------------------------------------

DEV_NAME = "Ahtasham Samad"
UPWORK_URL = "https://www.upwork.com/freelancers/~01d75561be7a2cd578"


def render_about_sidebar():
    with st.sidebar:
        st.divider()
        st.subheader("About")
        st.markdown(f"**Built by [{DEV_NAME}]({UPWORK_URL})** — AI/ML Developer")
        st.markdown("**Tools:** Python, Streamlit, scikit-learn (TF-IDF), "
                    "pypdf, python-docx, Plotly, fpdf2")


def render_footer():
    st.divider()
    st.markdown(
        f"Need a custom AI app? **[Hire me on Upwork]({UPWORK_URL})** · "
        f"Built by {DEV_NAME}"
    )


st.set_page_config(page_title="ATS Resume Screener", page_icon="📋",
                   layout="wide")

st.title("📋 ATS Resume Screener")
st.caption("⏳ If the app was asleep, it may take up to a minute to wake — "
           "please wait.")
st.info(
    "**What it does:** Ranks resumes against a job description — match score, "
    "skill gaps, section feedback and concrete fixes. No black box.\n\n"
    "**Who it's for:** Recruiters screening many applicants, and candidates "
    "who want to beat keyword-only ATS filters.\n\n"
    "**How to use:** ① Paste a job description (or load the sample) → "
    "② Upload resumes → ③ Rank candidates and download the report."
)
st.caption("🔒 **Privacy:** resumes are processed in memory only — they are "
           "never stored or sent anywhere.")
render_about_sidebar()

SKILLS = load_skills()

if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""
if "resume_files" not in st.session_state:
    st.session_state.resume_files = []  # list of (name, text)


def gauge_figure(score, name):
    """Plotly gauge visualising a candidate's overall ATS match score (0-100)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": f"Top match: {name}"},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#1f77b4"},
            "steps": [
                {"range": [0, 50], "color": "#f2f2f2"},
                {"range": [50, 75], "color": "#d9e8f5"},
                {"range": [75, 100], "color": "#c6e6c6"},
            ],
            "threshold": {"line": {"color": "#d62728", "width": 4},
                          "thickness": 0.75, "value": 80},
        },
    ))
    fig.update_layout(height=280, margin={"t": 60, "b": 20, "l": 20, "r": 20})
    return fig


def skill_pills(skills, kind):
    """Render skills as colored pill badges (HTML).

    kind="matched" -> green pills, kind="missing" -> red pills.
    Skill names are HTML-escaped, so this is safe to render.
    """
    if not skills:
        return "—"
    if kind == "matched":
        bg, fg = "#d4edda", "#155724"
    else:
        bg, fg = "#f8d7da", "#721c24"
    return " ".join(
        f'<span style="background:{bg};color:{fg};padding:2px 10px;'
        f'border-radius:12px;margin:2px;display:inline-block;'
        f'font-size:0.85em;">{html.escape(s)}</span>'
        for s in skills
    )


# ---------- sample data ----------
if st.button("✨ Load sample data (AI/ML Engineer posting + 5 resumes)"):
    jd, resumes = load_sample_data()
    st.session_state.jd_text = jd
    st.session_state.resume_files = resumes
    st.session_state.pop("ranked", None)
    st.success(f"Loaded job description + {len(resumes)} sample resumes.")


def merge_uploads(existing, new_files):
    """Append newly uploaded resumes; same filename replaces the old entry."""
    merged = list(existing)
    names = {n for n, _ in merged}
    for name, text in new_files:
        if name in names:
            merged = [(n, t) if n != name else (name, text)
                      for n, t in merged]
        else:
            merged.append((name, text))
            names.add(name)
    return merged


# ---------- inputs ----------
col_jd, col_res = st.columns(2)

with col_jd:
    st.subheader("1️⃣ Job description")
    jd_text = st.text_area(
        "Paste the job description", height=280, key="jd_text",
        help="Or upload it below — the upload replaces this text.")
    jd_upload = st.file_uploader("…or upload JD (.txt / .pdf / .docx)",
                                 type=["txt", "pdf", "docx"], key="jd_up")
    if jd_upload is not None:
        jd_text, jd_error = read_text_file(jd_upload)
        if jd_error:
            st.error(f"❌ {jd_upload.name}: {jd_error}")
        elif jd_text.strip():
            st.session_state.jd_text = jd_text
            st.success(f"Loaded JD from {jd_upload.name}.")
            st.rerun()

with col_res:
    st.subheader("2️⃣ Resumes")
    uploads = st.file_uploader(
        "Upload resumes (.pdf / .docx / .txt)", type=["pdf", "docx", "txt"],
        accept_multiple_files=True, key="res_up",
        help="New uploads are added to the list — existing ones are kept.")
    if uploads:
        files, problems = [], []
        for u in uploads:
            text, error = read_text_file(u)
            files.append((name_from_filename(u.name), text))
            if error:
                problems.append((u.name, error))
        st.session_state.resume_files = merge_uploads(
            st.session_state.resume_files, files)
        st.session_state.pop("ranked", None)
        for fname, error in problems:
            st.error(f"❌ {fname}: {error}")
        n_ok = len(files) - len(problems)
        if n_ok:
            st.success(f"Added {n_ok} resume(s) — total "
                       f"{len(st.session_state.resume_files)}.")
    if st.session_state.resume_files:
        names = [n for n, _ in st.session_state.resume_files]
        st.write(f"**{len(names)}** resume(s) ready: " + ", ".join(names))
        if st.button("🗑 Clear resumes"):
            st.session_state.resume_files = []
            st.session_state.pop("ranked", None)
            st.rerun()
        empty = [n for n, t in st.session_state.resume_files if not t.strip()]
        if empty:
            st.warning(
                "No extractable text in: " + ", ".join(empty)
                + " (scanned/image PDFs need OCR first).")

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
        st.success(f"Scored {len(ranked)} candidate(s).")

if st.session_state.get("ranked"):
    ranked = st.session_state.ranked
    st.subheader("📊 Ranked shortlist")

    top = ranked[0]
    g1, g2 = st.columns([1, 2])
    with g1:
        st.plotly_chart(gauge_figure(top["score"], top["name"]),
                        use_container_width=True)
    with g2:
        st.metric("Candidates scored", len(ranked))
        st.metric("Top TF-IDF similarity", f"{top['tfidf_similarity']:.3f}")
        st.metric("Top skill overlap", f"{top['skill_overlap']:.0%}")

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
            st.markdown(skill_pills(r["matched_skills"], "matched"),
                        unsafe_allow_html=True)
            st.markdown("**❌ Missing skills (wanted by the JD)**")
            st.markdown(skill_pills(r["missing_skills"], "missing"),
                        unsafe_allow_html=True)
            st.markdown("**📑 Section-wise feedback**")
            for section, (ok, note) in analyze_sections(r["text"]).items():
                icon = "✅" if ok else "⚠️"
                st.write(f"{icon} **{section}:** {note}")
            st.markdown("**💡 Suggestions to improve this resume**")
            for s in build_suggestions(r, jd_text):
                st.write(f"• {s}")
            st.info(r["summary"])

    st.caption(
        "Scores rank candidates for **shortlisting only** — they are similarity "
        "measurements, not hiring decisions. Always have a human review the shortlist."
    )

    # ---- downloads ----
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "⬇ Download ranking as CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="ats_ranking.csv", mime="text/csv")
    with d2:
        with st.spinner("Building PDF report…"):
            pdf_bytes = build_report_pdf(ranked, jd_text)
        st.download_button(
            "⬇ Download full report as PDF", pdf_bytes,
            file_name="ats_report.pdf", mime="application/pdf")

render_footer()
