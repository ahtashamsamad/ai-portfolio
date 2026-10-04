# 📊 Review Sentiment Dashboard

Classify customer reviews as positive, neutral or negative — with confidence scores, trends and keyword insights.

**Live demo:** https://ai-portfolio-flhuxakr6chyv53opzkcbx.streamlit.app/

## Screenshots

> Screenshots will be added after the next deploy (`docs/screenshots/`).

## Features

- **Single review** analysis with confidence score, or **batch CSV upload**
- **"Try with sample"** button — 30 bundled reviews, no upload needed
- Positive / neutral / negative classification (low-confidence predictions become *neutral* instead of forced calls)
- **Column selector** — pick which CSV column holds the review text
- **Filters:** sentiment multiselect + date-range filter (auto-detected date column)
- Interactive Plotly charts: distribution pie, **daily sentiment trend**, top positive/negative words, confidence histogram
- Summary metric cards: total reviews, % positive/neutral/negative, avg confidence
- Cached batch predictions (`st.cache_data`) for large files, **download results as CSV**
- Friendly error handling for bad/empty/oversized CSVs

## Tech stack

Python · Streamlit · scikit-learn (TF-IDF + LogisticRegression) · pandas · Plotly

## Run locally

```bash
git clone https://github.com/ahtashamsamad/ai-portfolio.git
cd ai-portfolio/review-sentiment-analyzer
pip install -r requirements.txt
streamlit run app.py
```

> Honest note: the demo model is trained on movie-style reviews — for a real business it should be retrained on that business's own reviews (`python generate_data.py && python train.py`).

## How it works

1. Reviews are vectorized with TF-IDF and classified by a LogisticRegression model (`train.py`).
2. Predictions below 60% confidence are labelled *neutral*.
3. Batch results are cached, aggregated into metrics and Plotly charts, and can be filtered by sentiment and date before downloading.

## Future improvements

- 5-star / aspect-based sentiment
- Retraining UI on the user's own labelled data
- Scheduled batch reports

## Author

**Ahtasham Samad** — AI/ML Developer
Upwork: https://www.upwork.com/freelancers/~01d75561be7a2cd578
