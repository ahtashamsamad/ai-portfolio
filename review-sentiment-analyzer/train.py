"""
Train a sentiment classifier: TF-IDF + LogisticRegression.

Reads reviews.csv (see generate_data.py), evaluates on a stratified held-out
split, prints accuracy / precision / recall / F1, and saves the whole
pipeline (vectorizer + classifier) to sentiment_model.joblib.
"""

import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "reviews.csv")
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.joblib")


def train(data_path=DATA_PATH, model_path=MODEL_PATH):
    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["sentiment"],
        test_size=0.2, random_state=42, stratify=df["sentiment"],
    )
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, pos_label="positive"),
        "recall": recall_score(y_test, pred, pos_label="positive"),
        "f1": f1_score(y_test, pred, pos_label="positive"),
        "n_test": len(y_test),
    }
    print(f"Test samples : {metrics['n_test']}")
    print(f"Accuracy     : {metrics['accuracy']:.3f}")
    print(f"Precision    : {metrics['precision']:.3f}")
    print(f"Recall       : {metrics['recall']:.3f}")
    print(f"F1           : {metrics['f1']:.3f}")

    joblib.dump(pipe, model_path)
    print(f"Model saved to {model_path}")
    return metrics


if __name__ == "__main__":
    train()
