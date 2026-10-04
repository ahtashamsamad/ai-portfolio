"""Customer Review Sentiment Dashboard (Streamlit).

- Type/paste a single review -> predicted label (positive / neutral /
  negative) + confidence.
- Upload a CSV of reviews -> every row classified, interactive Plotly
  charts (distribution pie, daily trend, top-words bars, confidence
  histogram), summary metric cards, sentiment + date filters, and a
  downloadable results CSV.
- "Try with sample" loads a bundled 30-review CSV — no upload needed.

The model is TF-IDF + LogisticRegression trained by train.py. If the saved
model file is missing (e.g. fresh deploy), the app trains it on the fly from
reviews.csv — the data is tiny, so this takes seconds.
"""

import os

import pandas as pd
import streamlit as st

from sentiment_backend import (
    MAX_BATCH_ROWS,
    NEG_WORDS,
    POS_WORDS,
    batch_predict,
    confidence_figure,
    detect_date_column,
    distribution_figure,
    get_model,
    predict_single,
    top_words,
    top_words_figure,
    trend_figure,
)

# ---------------------------------------------------------------------------
# Portfolio chrome
# ---------------------------------------------------------------------------

DEV_NAME = "Ahtasham Samad"
UPWORK_URL = "https://www.upwork.com/freelancers/~01d75561be7a2cd578"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_CSV_PATH = os.path.join(BASE_DIR, "sample_reviews.csv")


def render_about_sidebar():
    with st.sidebar:
        st.divider()
        st.subheader("About")
        st.markdown(f"**Built by [{DEV_NAME}]({UPWORK_URL})** — AI/ML Developer")
        st.markdown("**Tools:** Python, Streamlit, scikit-learn (TF-IDF + "
                    "LogisticRegression), Plotly, pandas")


def render_footer():
    st.divider()
    st.markdown(
        f"Need a custom AI app? **[Hire me on Upwork]({UPWORK_URL})** · "
        f"Built by {DEV_NAME}"
    )


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

@st.cache_resource
def _cached_model():
    return get_model()


@st.cache_data(show_spinner=False)
def cached_batch_predict(_model, texts):
    """Classify an immutable tuple of texts; cached across reruns.

    `_model` is underscore-prefixed so Streamlit skips hashing it; `texts`
    must be a tuple (hashable) for the cache key.
    """
    return batch_predict(_model, list(texts))


def load_sample_df():
    return pd.read_csv(SAMPLE_CSV_PATH)


def main():
    st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="📊",
                       layout="wide")
    st.title("📊 Review Sentiment Dashboard")
    st.caption("⏳ If the app was asleep, it may take up to a minute to wake — "
               "please wait.")
    st.info(
        "**What it does:** Classifies customer reviews as positive, neutral "
        "or negative — with confidence scores, trends and keyword insights.\n\n"
        "**Who it's for:** Businesses, product teams and marketers who want "
        "fast feedback analysis without reading every review.\n\n"
        "**How to use:** ① Paste a single review or upload a CSV (or try the "
        "sample) → ② Pick the text column → ③ Explore the charts and download "
        "the results."
    )
    st.caption("Honest note: this demo model is trained on movie-style reviews "
               "— for a real business it should be retrained on that "
               "business's own reviews.")
    render_about_sidebar()

    model = _cached_model()

    # ---- single review ----
    st.subheader("Single review")
    text = st.text_area("Paste a review", height=100)
    if st.button("Analyze") and text.strip():
        with st.spinner("Analyzing…"):
            label, confidence = predict_single(model, text)
        emoji = {"positive": "😊", "neutral": "😐", "negative": "😞"}[label]
        st.metric("Prediction", f"{emoji} {label}",
                  f"{confidence:.1%} confidence")

    # ---- batch ----
    st.subheader("Batch analysis")
    c_up, c_sample = st.columns([3, 1])
    with c_up:
        uploaded = st.file_uploader("Upload a CSV of reviews", type=["csv"])
    with c_sample:
        st.write("")
        if st.button("✨ Try with sample",
                     help="Load a bundled 30-review CSV — no upload needed."):
            st.session_state.sample_df = load_sample_df()

    df = None
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read that CSV ({exc}). "
                     "Make sure it is a valid CSV file.")
            render_footer()
            return
        if df.empty:
            st.warning("That CSV has no rows to analyze.")
            render_footer()
            return
    elif "sample_df" in st.session_state:
        df = st.session_state.sample_df
        st.info("Sample data loaded — 30 movie reviews with dates.")

    if df is None:
        st.info("Upload a CSV (or try the sample) to see batch analysis.")
        render_footer()
        return

    # ---- column selection ----
    text_cols = [c for c in df.columns
                 if df[c].dtype == object or str(df[c].dtype).startswith("str")]
    if not text_cols:
        st.error("No text column found in that CSV.")
        render_footer()
        return
    default_col = "review" if "review" in text_cols else text_cols[0]
    col = st.selectbox("Which column holds the review text?",
                       text_cols, index=text_cols.index(default_col))

    texts = df[col].astype(str).tolist()
    if len(texts) > MAX_BATCH_ROWS:
        st.warning(f"Large file — analyzing the first {MAX_BATCH_ROWS:,} of "
                   f"{len(texts):,} rows.")
        texts = texts[:MAX_BATCH_ROWS]
        df = df.head(MAX_BATCH_ROWS)

    with st.spinner(f"Classifying {len(texts):,} reviews…"):
        results = cached_batch_predict(model, tuple(texts))
    results["date_raw"] = df[detect_date_column(df)].values \
        if detect_date_column(df) else None

    # ---- filters ----
    f1, f2 = st.columns(2)
    with f1:
        chosen = st.multiselect(
            "Filter by sentiment", ["positive", "neutral", "negative"],
            default=["positive", "neutral", "negative"])
    date_col = detect_date_column(df)
    with f2:
        date_range = None
        if date_col:
            dates = pd.to_datetime(df[date_col], errors="coerce")
            dmin, dmax = dates.min().date(), dates.max().date()
            date_range = st.date_input("Date range", (dmin, dmax))
    filt = results[results["sentiment"].isin(chosen)].copy()
    if date_col and date_range and len(date_range) == 2:
        mask = (pd.to_datetime(filt["date_raw"], errors="coerce").dt.date
                >= date_range[0]) & \
               (pd.to_datetime(filt["date_raw"], errors="coerce").dt.date
                <= date_range[1])
        filt = filt[mask]
    if filt.empty:
        st.warning("No reviews match the current filters.")
        render_footer()
        return

    # ---- summary metric cards ----
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total reviews", f"{len(filt):,}")
    m2.metric("😊 Positive", f"{(filt['sentiment'] == 'positive').mean():.1%}")
    m3.metric("😐 Neutral", f"{(filt['sentiment'] == 'neutral').mean():.1%}")
    m4.metric("😞 Negative", f"{(filt['sentiment'] == 'negative').mean():.1%}")
    m5.metric("Avg confidence", f"{filt['confidence'].mean():.1%}")

    # ---- charts ----
    counts = filt["sentiment"].value_counts()
    st.plotly_chart(distribution_figure(counts), use_container_width=True)

    if date_col:
        dated = filt.assign(date=pd.to_datetime(filt["date_raw"],
                                                errors="coerce")).dropna(
            subset=["date"])
        fig = trend_figure(dated)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    st.plotly_chart(confidence_figure(filt["confidence"]),
                    use_container_width=True)

    pos = filt.loc[filt["sentiment"] == "positive", "review"]
    neg = filt.loc[filt["sentiment"] == "negative", "review"]
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

    # ---- results table + download ----
    with st.expander(f"📋 Classified reviews ({len(filt):,})"):
        st.dataframe(filt[["review", "sentiment", "confidence"]],
                     hide_index=True, use_container_width=True)
    csv_bytes = filt[["review", "sentiment", "confidence"]].to_csv(
        index=False).encode("utf-8")
    st.download_button("⬇ Download results as CSV", csv_bytes,
                       file_name="sentiment_results.csv", mime="text/csv")

    render_footer()


def _in_streamlit_runtime():
    try:
        from streamlit.runtime import exists
        return bool(exists())
    except Exception:
        return False


if _in_streamlit_runtime():
    main()
