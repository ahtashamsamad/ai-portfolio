"""Sentiment backend — pure functions, no Streamlit dependency.

Model loading/training, prediction, text analysis and Plotly figure
builders. The Streamlit UI lives in app.py.
"""


import io
import os
import re
from collections import Counter

import joblib
import pandas as pd
import plotly.express as px

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

MAX_BATCH_ROWS = 5000  # cap for a single CSV upload (keeps memory predictable)
LOW_CONFIDENCE = 0.60  # predictions below this become "neutral"

# Label shown when the model isn't confident enough to call it.
NEUTRAL_LABEL = "neutral"


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
    """Return (label, confidence) for one review string.

    Low-confidence predictions (< LOW_CONFIDENCE) are labelled "neutral"
    instead of forcing a positive/negative call.
    """
    raw_label = model.predict([text])[0]
    confidence = float(model.predict_proba([text]).max())
    label = raw_label if confidence >= LOW_CONFIDENCE else NEUTRAL_LABEL
    return label, confidence


def batch_predict(model, texts):
    """Return a DataFrame with each text, its label and confidence.

    Labels are positive / negative / neutral (neutral = low confidence).
    """
    raw_labels = model.predict(texts)
    confidences = model.predict_proba(texts).max(axis=1)
    labels = [lbl if conf >= LOW_CONFIDENCE else NEUTRAL_LABEL
              for lbl, conf in zip(raw_labels, confidences)]
    return pd.DataFrame({"review": texts, "sentiment": labels,
                         "confidence": confidences.round(3)})


def detect_date_column(df):
    """Return the first column that parses as datetimes (>=80% valid), else None."""
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            continue
        if parsed.notna().mean() >= 0.8:
            return col
    return None


def top_words(texts, lexicon, n=10):
    """Most frequent lexicon words across a list of review strings."""
    counter = Counter()
    for text in texts:
        for word in re.findall(r"[a-z]+", text.lower()):
            if word in lexicon:
                counter[word] += 1
    return counter.most_common(n)


def distribution_figure(counts):
    """Interactive Plotly pie chart of positive / neutral / negative counts."""
    order = ["positive", "neutral", "negative"]
    counts = counts.reindex([c for c in order if c in counts.index])
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        title="Sentiment distribution",
        color=counts.index,
        color_discrete_map={"positive": "#2ca02c", "neutral": "#9e9e9e",
                            "negative": "#d62728"},
    )
    fig.update_traces(textinfo="label+percent+value",
                      hoverinfo="label+value+percent")
    return fig


def trend_figure(dated):
    """Daily % positive/negative/neutral line chart (needs a date column).

    `dated` is a DataFrame with 'date' (datetime) and 'sentiment' columns.
    Returns None when there is nothing to plot.
    """
    if dated.empty:
        return None
    daily = (dated.assign(day=dated["date"].dt.date)
                  .groupby(["day", "sentiment"]).size()
                  .unstack(fill_value=0))
    daily = daily.div(daily.sum(axis=1), axis=0).mul(100).round(1)
    fig = px.line(
        daily.reset_index().melt(id_vars="day", var_name="sentiment",
                                 value_name="pct"),
        x="day", y="pct", color="sentiment",
        title="Sentiment trend (% of reviews per day)",
        labels={"day": "Date", "pct": "% of reviews"},
        color_discrete_map={"positive": "#2ca02c", "neutral": "#9e9e9e",
                            "negative": "#d62728"},
    )
    fig.update_layout(yaxis={"range": [0, 100]})
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


def confidence_figure(confidences):
    """Interactive Plotly histogram of prediction confidence scores."""
    fig = px.histogram(
        x=confidences,
        nbins=20,
        title="Confidence distribution",
        labels={"x": "Confidence", "y": "Reviews"},
        color_discrete_sequence=["#1f77b4"],
    )
    fig.add_vline(x=LOW_CONFIDENCE, line_dash="dash", line_color="red",
                  annotation_text="review threshold")
    return fig


# ---------------------------------------------------------------------------
