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
    """'ali_raza_resume.txt' -> 'Ali Raza'."""
    base = os.path.basename(filename)
    stem = re.sub(r"\.(txt|pdf)$", "", base, flags=re.IGNORECASE)
    stem = re.sub(r"[_\-]+", " ", stem)
    cleaned = re.sub(r"\b(resume|cv)\b", "", stem, flags=re.IGNORECASE).strip()
    name = " ".join(w.capitalize() for w in cleaned.split())
    return name or stem.strip()


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


def read_text_file(uploaded):
    """Extract text from an uploaded .txt or .pdf file.

    Returns (text, error) with error == "" on success. Never raises.
    """
    name = uploaded.name.lower()
    if not name.endswith(".pdf"):
        try:
            return uploaded.read().decode("utf-8", errors="ignore"), ""
        except Exception:
            return "", "could not be read as text"
    try:
        return extract_pdf_text(uploaded)
    except Exception:
        return "", "could not be read"


def load_sample_data(jd_path=SAMPLE_JD_PATH, resumes_dir=SAMPLE_RESUMES_DIR):
    """Load the bundled sample JD + resumes. Returns (jd_text, [(name, text)])."""
    with open(jd_path, encoding="utf-8") as f:
        jd = f.read()
    resumes = []
    for path in sorted(glob.glob(os.path.join(resumes_dir, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            resumes.append((name_from_filename(path), f.read()))
    return jd, resumes
