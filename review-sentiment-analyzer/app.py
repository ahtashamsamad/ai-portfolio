"""
Customer Review Sentiment Dashboard (Streamlit).

- Type or paste a single review -> predicted label + confidence.
- Upload a CSV of reviews -> every row classified, distribution bar chart,
  most frequent positive/negative words, and a downloadable results CSV.

The model is TF-IDF + LogisticRegression trained by train.py. If the saved
model file is missing (e.g. fresh deploy), the app trains it on the fly from
reviews.csv — the data is tiny, so this takes seconds.
"""

import pandas as pd
import streamlit as st

from sentiment_backend import (
    LOW_CONFIDENCE,
    MAX_BATCH_ROWS,
    NEG_WORDS,
    POS_WORDS,
    batch_predict,
    confidence_figure,
    distribution_figure,
    get_model,
    predict_single,
    top_words,
    top_words_figure,
)


# Streamlit UI
# ---------------------------------------------------------------------------

@st.cache_resource
def _cached_model():
    return get_model()


@st.cache_data(show_spinner=False)
def cached_batch_predict(_model, texts):
    """Classify an immutable tuple of texts; results are cached across reruns.

    `_model` is underscore-prefixed so Streamlit skips hashing it; `texts`
    must be a tuple (hashable) for the cache key.
    """
    return batch_predict(_model, list(texts))


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
        try:
            df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read that CSV ({exc}). "
                     "Make sure it is a valid CSV file.")
            return
        if df.empty:
            st.warning("That CSV has no rows to analyze.")
            return
        col = "review" if "review" in df.columns else df.columns[0]
        texts = df[col].astype(str).tolist()
        if len(texts) > MAX_BATCH_ROWS:
            st.warning(f"Large file — analyzing the first {MAX_BATCH_ROWS:,} of "
                       f"{len(texts):,} rows.")
            texts = texts[:MAX_BATCH_ROWS]
        st.caption(f"Classifying column: `{col}` ({len(texts):,} rows)")

        # Cached: repeat reruns (e.g. tweaking charts below) don't re-classify.
        with st.spinner("Classifying reviews…"):
            results = cached_batch_predict(model, tuple(texts))

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg confidence", f"{results['confidence'].mean():.1%}")
        c2.metric("High confidence (≥ 80%)",
                  f"{(results['confidence'] >= 0.8).mean():.1%}")
        c3.metric(f"Needs review (< {LOW_CONFIDENCE:.0%})",
                  f"{(results['confidence'] < LOW_CONFIDENCE).sum():,}")

        counts = results["sentiment"].value_counts()
        st.plotly_chart(distribution_figure(counts), use_container_width=True)
        st.plotly_chart(confidence_figure(results["confidence"]),
                        use_container_width=True)

        low = results[results["confidence"] < LOW_CONFIDENCE]
        if not low.empty:
            with st.expander(f"⚠️ {len(low)} low-confidence predictions "
                             f"(< {LOW_CONFIDENCE:.0%}) — worth a human look"):
                st.dataframe(low, hide_index=True, use_container_width=True)

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
