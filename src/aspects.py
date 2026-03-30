"""
aspects.py — Aspect-based sentiment extraction using spaCy dependency parsing

Instead of just "this review is positive/negative" (document-level sentiment),
aspect-based sentiment tells you WHAT people like or dislike:
    "The cushion is comfortable but assembly was terrible"
    → (cushion, comfortable, positive), (assembly, terrible, negative)

This uses spaCy's dependency parser to find adjective-noun pairs, then
scores the adjective's sentiment to get structured triples.
"""

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import spacy
from textblob import TextBlob

from preprocessing import load_and_clean_data



# aspect extraction
def extract_aspects(text, nlp):
    """
    Extract (aspect, opinion, sentiment) triples from a single review.

    Uses spaCy's dependency parser to find patterns like:
        - "comfortable cushion" → adjective modifying a noun (amod relation)
        - "quality is great" → adjective linked to noun via copula (nsubj + acomp)

    Then uses TextBlob to score the adjective's sentiment polarity.
    TextBlob returns a score from -1 (very negative) to +1 (very positive).

    Why dependency parsing instead of just finding all adjective-noun pairs?
    Because "not comfortable cushion" — the dependency tree shows "not" modifies
    "comfortable", so we can detect negation. Simple adjacency would miss this.
    """
    doc = nlp(text)
    aspects = []

    for token in doc:
        # Pattern 1: adjective directly modifying a noun
        # "comfortable cushion", "flimsy handle", "beautiful design"
        if token.dep_ == "amod" and token.head.pos_ == "NOUN":
            aspect = token.head.text
            opinion = token.text

            # check for negation — "not comfortable" should flip the sentiment
            negated = any(child.dep_ == "neg" for child in token.children)

            polarity = TextBlob(opinion).sentiment.polarity
            if negated:
                polarity *= -1

            sentiment = "positive" if polarity > 0 else "negative" if polarity < 0 else "neutral"
            aspects.append((aspect, opinion, sentiment, polarity))

        # Pattern 2: noun linked to adjective via copula ("is", "was", "seems")
        # "the quality is great", "the size was perfect"
        if token.dep_ == "nsubj" and token.pos_ == "NOUN":
            for child in token.head.children:
                if child.dep_ == "acomp" and child.pos_ == "ADJ":
                    aspect = token.text
                    opinion = child.text

                    negated = any(c.dep_ == "neg" for c in child.children)
                    polarity = TextBlob(opinion).sentiment.polarity
                    if negated:
                        polarity *= -1

                    sentiment = "positive" if polarity > 0 else "negative" if polarity < 0 else "neutral"
                    aspects.append((aspect, opinion, sentiment, polarity))

    return aspects


def extract_aspects_corpus(df, text_col="clean_text", batch_size=500):
    """
    Extract aspects from the full corpus.

    We use nlp.pipe() for speed, but this time we need the parser and
    tagger enabled (unlike preprocessing where we disabled them).
    This makes it slower — expect ~5-10 minutes on 50k reviews.
    """
    nlp = spacy.load("en_core_web_sm")

    print(f"Extracting aspects from {len(df):,} reviews...")

    all_aspects = []
    texts = df[text_col].tolist()

    for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size)):
        aspects = extract_aspects_from_doc(doc)
        all_aspects.append(aspects)

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1:,} reviews...")

    df = df.copy()
    df["aspects"] = all_aspects
    print(f"Done. Found aspects in {sum(1 for a in all_aspects if a):,} reviews")

    return df


def extract_aspects_from_doc(doc):
    """Same logic as extract_aspects() but takes a spaCy Doc directly."""
    aspects = []

    for token in doc:
        if token.dep_ == "amod" and token.head.pos_ == "NOUN":
            aspect = token.head.text
            opinion = token.text
            negated = any(child.dep_ == "neg" for child in token.children)
            polarity = TextBlob(opinion).sentiment.polarity
            if negated:
                polarity *= -1
            sentiment = "positive" if polarity > 0 else "negative" if polarity < 0 else "neutral"
            aspects.append((aspect, opinion, sentiment, polarity))

        if token.dep_ == "nsubj" and token.pos_ == "NOUN":
            for child in token.head.children:
                if child.dep_ == "acomp" and child.pos_ == "ADJ":
                    aspect = token.text
                    opinion = child.text
                    negated = any(c.dep_ == "neg" for c in child.children)
                    polarity = TextBlob(opinion).sentiment.polarity
                    if negated:
                        polarity *= -1
                    sentiment = "positive" if polarity > 0 else "negative" if polarity < 0 else "neutral"
                    aspects.append((aspect, opinion, sentiment, polarity))

    return aspects


# ============================================================================
# AGGREGATION
# ============================================================================

def aggregate_aspects(df):
    """
    Aggregate aspect-level sentiment across all reviews.

    For each aspect noun (e.g., "quality", "price", "size"), count how many
    times it was mentioned with a positive vs negative opinion. This gives
    you a product-level breakdown: "78% of mentions of 'quality' are positive."
    """
    aspect_data = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0, "total": 0})

    for aspects in df["aspects"]:
        for aspect, opinion, sentiment, polarity in aspects:
            aspect_data[aspect][sentiment] += 1
            aspect_data[aspect]["total"] += 1

    # convert to DataFrame and sort by frequency
    rows = []
    for aspect, counts in aspect_data.items():
        if counts["total"] >= 10:  # filter out rare aspects
            rows.append({
                "aspect": aspect,
                "total": counts["total"],
                "positive": counts["positive"],
                "negative": counts["negative"],
                "neutral": counts["neutral"],
                "pct_positive": counts["positive"] / counts["total"] * 100,
            })

    agg_df = pd.DataFrame(rows).sort_values("total", ascending=False).reset_index(drop=True)
    return agg_df


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_top_aspects(agg_df, top_n=20, save_path="figures/aspect_sentiment.png"):
    """Plot top aspects by frequency with sentiment breakdown."""
    top = agg_df.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 8))

    y = range(len(top))
    ax.barh(y, top["positive"], color="#2ecc71", label="Positive")
    ax.barh(y, top["negative"], left=top["positive"], color="#e74c3c", label="Negative")
    ax.barh(y, top["neutral"], left=top["positive"] + top["negative"],
            color="#95a5a6", label="Neutral")

    ax.set_yticks(y)
    ax.set_yticklabels(top["aspect"])
    ax.set_xlabel("Number of Mentions")
    ax.set_title(f"Top {top_n} Aspects by Frequency with Sentiment Breakdown")
    ax.legend()
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    df = load_and_clean_data()
    if df is None:
        raise RuntimeError("Run preprocessing.py first")

    # ------------------------------------------------------------------
    # 1. EXTRACT ASPECTS
    # ------------------------------------------------------------------

    nlp = spacy.load("en_core_web_sm")

    # demo on a few examples first
    print("=== ASPECT EXTRACTION EXAMPLES ===\n")
    examples = [
        "the cushion is very comfortable but the legs are flimsy",
        "great quality for the price, easy to assemble",
        "terrible product, broke after two days, not worth the money",
        "beautiful design but the material feels cheap",
    ]
    for text in examples:
        aspects = extract_aspects(text, nlp)
        print(f"Text: {text}")
        for a, o, s, p in aspects:
            print(f"  → ({a}, {o}, {s}, polarity={p:.2f})")
        print()

    # ------------------------------------------------------------------
    # 2. EXTRACT FROM FULL CORPUS
    # ------------------------------------------------------------------

    df = extract_aspects_corpus(df)

    # save for use by other scripts
    df.to_parquet("data/reviews_with_aspects.parquet")

    # ------------------------------------------------------------------
    # 3. AGGREGATE & VISUALIZE
    # ------------------------------------------------------------------

    agg_df = aggregate_aspects(df)

    print("\n=== TOP ASPECTS ===")
    print(agg_df.head(20).to_string(index=False))

    plot_top_aspects(agg_df)

    # show most positive and most negative aspects
    frequent = agg_df[agg_df["total"] >= 50]

    print("\nMost Positive Aspects (min 50 mentions):")
    most_pos = frequent.nlargest(10, "pct_positive")
    for _, row in most_pos.iterrows():
        print(f"  {row['aspect']:20s} {row['pct_positive']:.0f}% positive ({row['total']} mentions)")

    print("\nMost Negative Aspects (min 50 mentions):")
    most_neg = frequent.nsmallest(10, "pct_positive")
    for _, row in most_neg.iterrows():
        print(f"  {row['aspect']:20s} {row['pct_positive']:.0f}% positive ({row['total']} mentions)")

    print("\n\nDone. Aspects saved to data/reviews_with_aspects.parquet")
    print("Next: run topics.py")