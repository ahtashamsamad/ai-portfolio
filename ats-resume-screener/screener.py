"""Core logic for the ATS Resume Screener.

Kept separate from app.py so it can be tested headless.

Scoring (0-100):
    score = 0.6 * TF-IDF cosine similarity (JD vs resume wording)
          + 0.4 * skill overlap (share of the JD's listed skills found in the resume)

Skill matching is case-insensitive and expands common abbreviations
("ML" -> "machine learning") with word-boundary-safe regex, so "ML"
matches but "HTML" is left untouched.
"""

import glob
import os
import re

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SKILLS_PATH = os.path.join(os.path.dirname(__file__), "skills.txt")

# Abbreviations mapped to their canonical skill names. Applied with word
# boundaries so "ML" matches but "HTML" is untouched, "AI" matches but
# words merely containing "ai" do not.
ALIASES = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "dl": "deep learning",
    "cv": "computer vision",
    "k8s": "kubernetes",
    "tf": "tensorflow",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "js": "javascript",
    "ts": "typescript",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "gcp": "google cloud",
    "genai": "generative ai",
    "llms": "llm",
}


def load_skills(path=SKILLS_PATH):
    """Load the skill vocabulary, one skill per line."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def normalize_text(text):
    """Lowercase + expand known abbreviations (word-boundary safe)."""
    text = text.lower()
    for alias, canonical in ALIASES.items():
        text = re.sub(r"\b" + re.escape(alias) + r"\b", canonical, text)
    return text


def extract_skills(text, skills):
    """Return the subset of the skill vocabulary mentioned in text."""
    normalized = normalize_text(text)
    found = []
    for skill in skills:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, normalized):
            found.append(skill)
    return found


def build_summary(matched, missing, similarity):
    """2-3 line plain-English explanation of a candidate's score."""
    lines = []
    if matched:
        lines.append("Overlaps with the posting on: " + ", ".join(matched[:5]) + ".")
    else:
        lines.append("No overlap with the posting's listed skills.")
    if missing:
        lines.append("Gaps vs the posting: " + ", ".join(missing[:5]) + ".")
    else:
        lines.append("Covers every skill the posting lists.")
    if similarity >= 0.5:
        lines.append(f"Overall wording is very close to the posting (similarity {similarity:.0%}).")
    elif similarity >= 0.25:
        lines.append(f"Overall wording is partially aligned with the posting (similarity {similarity:.0%}).")
    else:
        lines.append(f"Overall wording differs substantially from the posting (similarity {similarity:.0%}).")
    return " ".join(lines)


def rank_resumes(jd_text, resumes, skills=None):
    """Rank resumes against a job description.

    jd_text: str — the job description.
    resumes: list of {"name": str, "text": str}.
    Returns a list of dicts sorted by score (desc), each with:
      name, score (0-100), tfidf_similarity, skill_overlap,
      jd_skills, matched_skills, missing_skills, summary.
    """
    skills = skills if skills is not None else load_skills()
    jd_skills = extract_skills(jd_text, skills)

    documents = [jd_text] + [r["text"] for r in resumes]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(documents)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    ranked = []
    for resume, sim in zip(resumes, sims):
        resume_skills = extract_skills(resume["text"], skills)
        matched = [s for s in jd_skills if s in resume_skills]
        missing = [s for s in jd_skills if s not in resume_skills]
        overlap = len(matched) / len(jd_skills) if jd_skills else 0.0
        score = round(0.6 * float(sim) * 100 + 0.4 * overlap * 100, 1)
        score = max(0.0, min(100.0, score))
        ranked.append(
            {
                "name": resume["name"],
                "text": resume["text"],
                "score": score,
                "tfidf_similarity": round(float(sim), 3),
                "skill_overlap": round(overlap, 3),
                "jd_skills": jd_skills,
                "matched_skills": matched,
                "missing_skills": missing,
                "summary": build_summary(matched, missing, float(sim)),
            }
        )
    ranked.sort(key=lambda r: r["score"], reverse=True)
    return ranked


def name_from_filename(filename):
    """'ali_raza_resume.txt' -> 'Ali Raza'; 'sample_resume.pdf' -> 'Sample Resume'."""
    base = os.path.basename(filename)
    stem = re.sub(r"\.(txt|pdf|docx)$", "", base, flags=re.IGNORECASE)
    stem = re.sub(r"[_\-]+", " ", stem).strip()
    cleaned = re.sub(r"\b(resume|cv)\b", "", stem, flags=re.IGNORECASE).strip()
    # Keep "resume"/"cv" when it was the only meaningful word.
    core = cleaned if len(cleaned.split()) >= 2 else stem
    name = " ".join(w.capitalize() for w in core.split())
    return name or stem


# ---------------------------------------------------------------------------
# File I/O helpers (used by the Streamlit UI in app.py; testable headless)
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_JD_PATH = os.path.join(BASE_DIR, "sample_jd.txt")
SAMPLE_RESUMES_DIR = os.path.join(BASE_DIR, "resumes")


def extract_pdf_text(uploaded):
    """Extract text from a PDF, trying layout mode first.

    Layout mode keeps multi-column resumes readable; plain mode is the
    fallback. Returns (text, error) with error == "" on success — one bad
    file never raises, so a single corrupt PDF can't break the whole batch.
    """
    last_error = ""
    for mode in ("layout", "plain"):
        try:
            uploaded.seek(0)
            reader = PdfReader(uploaded)
            if reader.is_encrypted:
                return "", "password-protected (decrypt it first)"
            text = "\n".join(
                (page.extract_text(extraction_mode=mode) or "")
                for page in reader.pages
            )
            if text.strip():
                return text, ""
            last_error = "no extractable text (scanned/image PDFs need OCR first)"
        except Exception as exc:
            last_error = f"could not be read ({type(exc).__name__})"
    return "", last_error


def extract_docx_text(uploaded):
    """Extract text from a .docx file. Returns (text, error); never raises."""
    try:
        from docx import Document
        uploaded.seek(0)
        doc = Document(uploaded)
        text = "\n".join(p.text for p in doc.paragraphs)
        if not text.strip():
            return "", "no extractable text in this Word document"
        return text, ""
    except Exception as exc:
        return "", f"could not be read ({type(exc).__name__})"


def read_text_file(uploaded):
    """Extract text from an uploaded .txt, .pdf or .docx file.

    Returns (text, error) with error == "" on success. Never raises.
    """
    name = uploaded.name.lower()
    if name.endswith(".pdf"):
        try:
            return extract_pdf_text(uploaded)
        except Exception:
            return "", "could not be read"
    if name.endswith(".docx"):
        return extract_docx_text(uploaded)
    if name.endswith(".txt"):
        try:
            uploaded.seek(0)
            return uploaded.read().decode("utf-8", errors="ignore"), ""
        except Exception:
            return "", "could not be read as text"
    return "", "unsupported file type (use .pdf, .docx or .txt)"


def load_sample_data(jd_path=SAMPLE_JD_PATH, resumes_dir=SAMPLE_RESUMES_DIR):
    """Load the bundled sample JD + resumes. Returns (jd_text, [(name, text)])."""
    with open(jd_path, encoding="utf-8") as f:
        jd = f.read()
    resumes = []
    for path in sorted(glob.glob(os.path.join(resumes_dir, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            resumes.append((name_from_filename(path), f.read()))
    return jd, resumes


# ---------------------------------------------------------------------------
# Section-wise feedback, suggestions and PDF report
# ---------------------------------------------------------------------------

SECTION_PATTERNS = {
    "skills": [r"\bskills?\b", r"\btechnologies\b", r"\btech stack\b",
               r"\bcompetencies\b"],
    "experience": [r"\bexperience\b", r"\bwork history\b", r"\bemployment\b",
                   r"\bprofessional background\b"],
    "education": [r"\beducation\b", r"\bbachelor", r"\bmaster'?s?\b", r"\bph\.?d\b",
                  r"\buniversity\b", r"\bcollege\b", r"\bdegree\b"],
}

ACTION_VERBS = {"built", "led", "designed", "developed", "launched",
                "improved", "increased", "reduced", "managed", "created",
                "implemented", "optimized", "delivered", "architected"}


def _has_section(text, patterns):
    low = text.lower()
    return any(re.search(p, low) for p in patterns)


def analyze_sections(text):
    """Heuristic section-by-section review of a resume.

    Returns an ordered dict: section -> (ok: bool, note: str).
    """
    low = text.lower()
    words = text.split()
    sections = {}

    # Skills
    n_skills = len(extract_skills(text, load_skills()))
    sections["Skills"] = (
        (n_skills >= 3, f"{n_skills} recognized skills detected.")
        if _has_section(text, SECTION_PATTERNS["skills"]) or n_skills >= 3
        else (False, "No clear Skills section found — add one listing your "
                     "key technologies."))

    # Experience
    years = bool(re.search(r"\d+\+?\s*(years|yrs)", low))
    verbs = sum(1 for v in ACTION_VERBS if re.search(r"\b" + v + r"\b", low))
    quantified = bool(re.search(r"\d+\s*%|\$\s*\d|\d+\s*(users|clients|requests)",
                                low))
    if _has_section(text, SECTION_PATTERNS["experience"]):
        note = []
        note.append("experience section present")
        note.append("years mentioned" if years else "no years-of-experience stated")
        note.append(f"{verbs} action verbs used")
        note.append("achievements quantified" if quantified
                    else "achievements not quantified — add numbers")
        sections["Experience"] = (verbs >= 2 and quantified, "; ".join(note) + ".")
    else:
        sections["Experience"] = (
            False, "No Experience section found — list roles with dates, "
                   "responsibilities and results.")

    # Education
    sections["Education"] = (
        (True, "education details found.")
        if _has_section(text, SECTION_PATTERNS["education"])
        else (False, "No Education section found — add your degree, field "
                     "and institution."))

    # Formatting & contact
    email = bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text))
    phone = bool(re.search(r"\+?\d[\d\s\-()]{7,}\d", text))
    bullets = text.count("•") + text.count("- ") + len(
        re.findall(r"^[\*\-]\s", text, re.M))
    n_words = len(words)
    fmt_notes = []
    fmt_notes.append("contact info present" if (email or phone)
                     else "no email/phone found — add contact details")
    fmt_notes.append(f"{n_words} words")
    if n_words < 150:
        fmt_notes.append("quite short — flesh out achievements")
        fmt_ok = False
    elif n_words > 1200:
        fmt_notes.append("very long — trim to the most relevant 1–2 pages")
        fmt_ok = False
    else:
        fmt_ok = bool(email or phone)
    if bullets < 3:
        fmt_notes.append("few bullet points — use bullets for readability")
        fmt_ok = False
    sections["Formatting"] = (fmt_ok, "; ".join(fmt_notes) + ".")
    return sections


def build_suggestions(candidate, jd_text):
    """3–5 concrete, actionable resume improvements for one candidate.

    candidate: a dict from rank_resumes(); jd_text: the job description.
    """
    suggestions = []
    text = candidate.get("text", "") or ""
    missing = candidate.get("missing_skills", [])

    if missing:
        suggestions.append(
            "Add these posting keywords your resume is missing: "
            + ", ".join(missing[:6]) + ".")
    if candidate.get("tfidf_similarity", 0) < 0.3:
        suggestions.append(
            "Mirror the posting's language — rephrase your experience using "
            "the same terms the job description uses.")
    low = text.lower()
    if not re.search(r"\d+\s*%|\$\s*\d|\d+\s*(users|clients|requests|projects)",
                     low):
        suggestions.append(
            "Quantify achievements with numbers (%, users, revenue, latency) "
            "— measurable impact stands out to recruiters and ATS alike.")
    if not _has_section(text, SECTION_PATTERNS["education"]):
        suggestions.append(
            "Add an Education section (degree, field, institution, year).")
    if len(text.split()) < 200:
        suggestions.append(
            "The resume is short — expand each role with 2–3 bullet points "
            "covering what you did and the result.")
    if not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text):
        suggestions.append(
            "Add contact details (email, phone, location/LinkedIn) at the top.")
    # Always useful, kept last so the list never comes back empty.
    suggestions.append(
        "Tailor this resume per application: keep the strongest 60–70% of "
        "content aligned to each specific posting.")
    return suggestions[:5]


def _latin1(text):
    """fpdf2's core fonts are latin-1 — strip anything else."""
    return text.encode("latin-1", errors="ignore").decode("latin-1")


def build_report_pdf(ranked, jd_text):
    """Build a PDF screening report. Returns bytes (fpdf2)."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, _latin1("ATS Resume Screening Report"), new_x="LMARGIN",
             new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, _latin1(f"Candidates scored: {len(ranked)}"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _latin1("Ranking"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for i, r in enumerate(ranked, 1):
        pdf.cell(0, 7, _latin1(
            f"#{i}  {r['name']}  —  {r['score']}% match  "
            f"(similarity {r['tfidf_similarity']:.2f}, "
            f"skills {r['skill_overlap']:.0%})"),
            new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    for i, r in enumerate(ranked, 1):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, _latin1(f"#{i} — {r['name']} ({r['score']}% match)"),
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _latin1(r["summary"]))
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, _latin1("Matched skills"), new_x="LMARGIN",
                 new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _latin1(", ".join(r["matched_skills"])
                                     or "None"))
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, _latin1("Missing skills"), new_x="LMARGIN",
                 new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _latin1(", ".join(r["missing_skills"])
                                     or "None"))
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, _latin1("Suggestions"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for s in build_suggestions(r, jd_text):
            pdf.multi_cell(0, 6, _latin1("- " + s))
    return bytes(pdf.output())
