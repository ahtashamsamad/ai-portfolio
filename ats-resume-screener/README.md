Live demo: _(coming soon)_

# 📋 ATS Resume Screener

HR teams receive hundreds of resumes per posting, and standard keyword-based
ATS (Applicant Tracking Systems) reject strong candidates over vocabulary
mismatch — "ML" vs "machine learning" shouldn't decide a career. This app
ranks candidates by **semantic similarity plus skill overlap** and shows
exactly *why* each candidate scored the way they did.

## Features

- Paste a job description (or upload it as `.txt` / `.pdf`)
- Upload multiple resumes (`.pdf` via `pypdf` text extraction, or `.txt`)
- One click → **ranked shortlist**: candidate name, match score %, matched /
  missing skill counts
- Per-candidate detail view: score bar, TF-IDF similarity, skill-overlap %,
  matched ✅ / missing ❌ skills, and a 2–3 line plain-English "why this
  candidate fits / gaps" summary
- Built-in skill vocabulary (`skills.txt`, ~125 tech + business skills) with
  abbreviation expansion (`ML` → `machine learning`, `NLP` → `natural
  language processing`), so candidates aren't punished for vocabulary mismatch
- **✨ Load sample data** button: an AI/ML Engineer posting + 5 sample resumes
  (2 strong, 2 partial, 1 weak) for an instant demo

## How scoring works (explainable, no black box)

```
score = 0.6 × TF-IDF cosine similarity (JD vs resume wording)
      + 0.4 × skill overlap (share of the JD's listed skills found in the resume)
```
scaled to 0–100. Skill matching is case-insensitive with word-boundary-safe
abbreviation expansion. Everything the score is made of is shown in the UI.

## Honest limitations

- Scores are **similarity-based rankings for shortlisting only** — they are
  measurements, not hiring decisions. A human must always review the shortlist.
- Simple keyword matching can't judge experience *depth* ("used TensorFlow
  once" vs "shipped TensorFlow to production" score the same).
- Short tokens can misfire (e.g. the verb "go" matching the **Go** language) —
  the skill list is a starting point, not a taxonomy.
- Scanned/image PDFs contain no extractable text; they need OCR first (the app
  warns you when a resume yields no text).

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Sample-data walkthrough

1. Click **✨ Load sample data**.
2. Click **🚀 Rank candidates**.
3. Expected result: the two AI/ML engineers (Ali Raza, Sara Khan) rank at the
   top, the data analyst and backend developer land in the middle, and the
   sales executive ranks last — each with matched/missing skills explained.

## Project structure

```
ats-resume-screener/
├── app.py            # Streamlit UI
├── screener.py       # scoring + skill extraction (testable headless)
├── skills.txt        # skill vocabulary (~125 skills)
├── sample_jd.txt     # sample job posting (AI/ML Engineer)
├── resumes/          # 5 sample resumes (.txt)
├── requirements.txt
└── README.md
```
