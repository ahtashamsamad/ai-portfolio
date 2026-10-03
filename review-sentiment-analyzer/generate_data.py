"""
Generate 400 labeled movie-style reviews (200 positive / 200 negative).

Design note (important for generalization): sentiment-bearing words are drawn
from small pools that REPEAT across many samples, while neutral context words
(movie/film/picture, acting/plot/…) vary. If every sample used unique wording,
the train/test split would share no vocabulary and the model could not
generalize — a classic demo-data mistake.

Output: reviews.csv with columns [review, sentiment].
"""

import csv
import os
import random

random.seed(7)  # reproducible dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE_DIR, "reviews.csv")

# --- Sentiment vocabularies: small pools, reused across MANY samples --------
POS_ADJ = ["excellent", "amazing", "fantastic", "wonderful", "brilliant",
           "outstanding", "superb", "delightful", "incredible", "phenomenal"]
POS_VERB = ["loved", "enjoyed", "adored"]
POS_NOUN = ["masterpiece", "gem"]

NEG_ADJ = ["terrible", "awful", "horrible", "boring", "disappointing",
           "dreadful", "unwatchable", "painful", "pathetic", "atrocious"]
NEG_VERB = ["hated", "disliked", "regretted"]
NEG_NOUN = ["disaster", "mess"]

# --- Neutral context vocabularies: varied freely, carry no sentiment --------
NOUNS = ["movie", "film", "picture", "flick", "show", "story"]
ASPECTS = ["acting", "plot", "cinematography", "soundtrack", "direction",
           "script", "pacing", "ending", "dialogue", "visuals"]
NEUTRAL_TAILS = ["The theater was packed.",
                 "It runs for about two hours.",
                 "I watched it on Friday night.",
                 "The tickets were reasonably priced.",
                 "My friend came along with me."]

POS_TEMPLATES = [
    "The {noun} was {adj} and I {verb} every minute of it.",
    "What an {adj} {noun}! The {aspect} was absolutely {adj2}.",
    "I {verb} this {noun}. {Adj} {aspect} and a truly {adj2} story.",
    "An {adj} {noun} from start to finish — the {aspect} alone is worth it.",
    "This {noun} is a real {pnoun}. {Adj} {aspect} and {adj2} pacing.",
    "I cannot recommend this {noun} enough. {Adj} {aspect}, and I {verb} the ending.",
    "The {aspect} was {adj} and the whole {noun} felt {adj2}.",
    "A {adj} {noun} with {adj2} {aspect}. I {verb} it.",
]

NEG_TEMPLATES = [
    "The {noun} was {adj} and I {verb} every minute of it.",
    "What an {adj} {noun}. The {aspect} was absolutely {adj2}.",
    "I {verb} this {noun}. {Adj} {aspect} and a truly {adj2} story.",
    "An {adj} {noun} from start to finish — the {aspect} alone was {adj2}.",
    "This {noun} is a real {pnoun}. {Adj} {aspect} and {adj2} pacing.",
    "I cannot recommend this {noun}. {Adj} {aspect}, and I {verb} the ending.",
    "The {aspect} was {adj} and the whole {noun} felt {adj2}.",
    "A {adj} {noun} with {adj2} {aspect}. I {verb} it.",
]


def make_review(templates, adjs, verbs, nouns_extra):
    t = random.choice(templates)
    adj, adj2 = random.sample(adjs, 2)
    text = t.format(
        noun=random.choice(NOUNS),
        adj=adj,
        adj2=adj2,
        verb=random.choice(verbs),
        aspect=random.choice(ASPECTS),
        pnoun=random.choice(nouns_extra),
        Adj=adj.capitalize(),
    )
    if random.random() < 0.5:  # neutral filler: adds vocab variety, no sentiment
        text += " " + random.choice(NEUTRAL_TAILS)
    return text


def main():
    rows = []
    for _ in range(200):
        rows.append((make_review(POS_TEMPLATES, POS_ADJ, POS_VERB, POS_NOUN), "positive"))
    for _ in range(200):
        rows.append((make_review(NEG_TEMPLATES, NEG_ADJ, NEG_VERB, NEG_NOUN), "negative"))
    random.shuffle(rows)

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["review", "sentiment"])
        writer.writerows(rows)

    pos = sum(1 for _, s in rows if s == "positive")
    print(f"Wrote {len(rows)} reviews to {OUT_PATH} ({pos} positive, {len(rows)-pos} negative)")


if __name__ == "__main__":
    main()
