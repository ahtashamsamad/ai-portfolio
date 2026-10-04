Live demo: https://ai-portfolio-flhuxakr6chyv53opzkcbx.streamlit.app/

# 📊 Review Sentiment Analyzer

Classifies customer reviews as positive or negative. Type a single review for
an instant prediction, or upload a CSV for batch analysis with a distribution
chart, the most frequent praise/complaint words, and a downloadable results
file.

**Honest note:** the demo model is trained on 400 synthetic movie-style
reviews (see `generate_data.py`). For a real business it should be retrained
on that business's own labeled reviews — the pipeline (`train.py`) makes that
a one-command job.

## Features

- **Single review**: instant prediction with confidence score
- **Batch CSV**: upload a whole file — validated on read (bad CSV / empty
  file shows a friendly error), capped at 5,000 rows per run, with a live
  progress bar
- `@st.cache_data` on batch predictions — tweaking charts below doesn't
  re-run the model
- **Interactive Plotly charts**: sentiment share (pie), top
  praise/complaint words (bars), prediction-confidence histogram
- **Confidence metrics**: average confidence, high-confidence share
  (≥ 80%), and a low-confidence review queue (< 60%) for human spot-checks
- Downloadable results CSV (review, sentiment, confidence)

## Measured results

From the actual `train.py` run (stratified 80/20 split, seed 42):

| Metric | Score |
|---|---|
| Accuracy | 1.000 |
| Precision | 1.000 |
| Recall | 1.000 |
| F1 | 1.000 |
| Held-out test samples | 80 |

The near-perfect score is expected: the synthetic dataset reuses a small pool
of sentiment-bearing words across many samples by design, so the classes are
cleanly separable. On messy real-world reviews the numbers will be lower —
which is exactly why retraining on real data matters.

## Run it

```bash
pip install -r requirements.txt
python generate_data.py   # builds reviews.csv (400 labeled reviews)
python train.py           # trains + saves sentiment_model.joblib, prints metrics
streamlit run app.py      # dashboard at http://localhost:8501
```

If `sentiment_model.joblib` is missing when the app starts (e.g. a fresh
deploy), the app trains it automatically from `reviews.csv`.

## Deploying (Streamlit Community Cloud)

Push this folder to GitHub, then at share.streamlit.io → New app → pick the
repo → set **Main file path** to `review-sentiment-analyzer/app.py` → Deploy.

## Project structure

```
review-sentiment-analyzer/
├── app.py               # Streamlit UI (cached model + predictions)
├── sentiment_backend.py # model loading, prediction, analysis, charts (headless)
├── train.py             # one-command training, prints metrics
├── generate_data.py     # builds the synthetic reviews.csv
├── reviews.csv          # 400 labeled reviews
├── requirements.txt
└── README.md
```
