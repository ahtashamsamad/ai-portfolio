"""
Customer Review Sentiment Dashboard (Streamlit).

- Type or paste a single review -> predicted label + confidence.
- Upload a CSV of reviews -> every row classified, distribution bar chart,
  most frequent positive/negative words, and a downloadable results CSV.

The model is TF-IDF + LogisticRegression trained by train.py. If the saved
model file is missing (e.g. fresh deploy), the app trains it on the fly from
reviews.csv — the data is tiny, so this takes seconds.
"""

import io
import os
import re
from collections import Counter

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "reviews.csv")
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.joblib")

# Sentiment lexicons used for the "top words" analysis (same vocabulary the
# dataset in generate_data.py is built from).
POS_WORDS = {"excellent", "amazing", "fantastic", "wonderful", "brilliant",
             "outstanding", "superb", "delightful", "incredible", "phenomenal",
             "loved", "enjoyed", "adored", "masterpiece", "gem"}
NEG_WORDS = {"terrible", "awful", "horrible", "boring", "disappointing",
             "dreadful", "unwatchable", "painful", "pathetic", "atrocious",
             "hated", "disliked", "regretted", "disaster", "mess"}


# ---------------------------------------------------------------------------
# Core logic (pure functions — easy to test, no Streamlit dependency)
# ---------------------------------------------------------------------------

def get_model():
    """Load the trained pipeline; train from reviews.csv if no model file yet."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    from train import train  # local import: only needed for the fallback path
    train(DATA_PATH, MODEL_PATH)
    return joblib.load(MODEL_PATH)


def predict_single(model, text):
    """Return (label, confidence) for one review string."""
    label = model.predict([text])[0]
    confidence = float(model.predict_proba([text]).max())
    return label, confidence


def batch_predict(model, texts):
    """Return a DataFrame with each text, its label and confidence."""
    labels = model.predict(texts)
    confidences = model.predict_proba(texts).max(axis=1)
    return pd.DataFrame({"review": texts, "sentiment": labels,
                         "confidence": confidences.round(3)})


def top_words(texts, lexicon, n=10):
    """Most frequent lexicon words across a list of review strings."""
    counter = Counter()
    for text in texts:
        for word in re.findall(r"[a-z]+", text.lower()):
            if word in lexicon:
                counter[word] += 1
    return counter.most_common(n)


def distribution_figure(counts):
    """Interactive Plotly pie chart of positive vs negative counts."""
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        title="Sentiment distribution",
        color=counts.index,
        color_discrete_map={"positive": "#2ca02c", "negative": "#d62728"},
    )
    fig.update_traces(textinfo="label+percent+value",
                      hoverinfo="label+value+percent")
    return fig


def top_words_figure(word_counts, title, color):
    """Interactive Plotly horizontal bar chart of the most frequent words.

    Returns None when there is nothing to plot.
    """
    if not word_counts:
        return None
    words, freq = zip(*word_counts)
    fig = px.bar(
        x=list(freq),
        y=list(words),
        orientation="h",
        title=title,
        labels={"x": "Mentions", "y": ""},
        color_discrete_sequence=[color],
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

@st.cache_resource
def _cached_model():
    return get_model()


def main():
    st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="📊")
    st.title("📊 Customer Review Sentiment Dashboard")
    st.caption("Classifies reviews as positive or negative. Honest note: this "
               "demo model is trained on movie-style reviews — for a real "
               "business it should be retrained on that business's own reviews.")

    model = _cached_model()

    st.subheader("Single review")
    text = st.text_area("Paste a review", height=100)
    if st.button("Analyze") and text.strip():
        label, confidence = predict_single(model, text)
        emoji = "😊" if label == "positive" else "😞"
        st.metric("Prediction", f"{emoji} {label}", f"{confidence:.1%} confidence")

    st.subheader("Batch analysis (CSV upload)")
    uploaded = st.file_uploader("Upload a CSV with a review-text column", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        col = "review" if "review" in df.columns else df.columns[0]
        st.caption(f"Classifying column: `{col}` ({len(df)} rows)")
        with st.spinner("Classifying…"):
            results = batch_predict(model, df[col].astype(str).tolist())

        counts = results["sentiment"].value_counts()
        st.plotly_chart(distribution_figure(counts), use_container_width=True)

        pos = results.loc[results["sentiment"] == "positive", "review"]
        neg = results.loc[results["sentiment"] == "negative", "review"]
        col1, col2 = st.columns(2)
        with col1:
            fig = top_words_figure(top_words(pos, POS_WORDS),
                                   "Top positive words", "#2ca02c")
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No positive lexicon words found.")
        with col2:
            fig = top_words_figure(top_words(neg, NEG_WORDS),
                                   "Top negative words", "#d62728")
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("No negative lexicon words found.")

        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download results as CSV", csv_bytes,
                           file_name="sentiment_results.csv", mime="text/csv")


def _in_streamlit_runtime():
    """True only when executed via `streamlit run` (lets us import headlessly)."""
    try:
        from streamlit.runtime import exists
        return bool(exists())
    except Exception:
        return False


if _in_streamlit_runtime():
    main()
