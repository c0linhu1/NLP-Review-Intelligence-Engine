"""
topics.py — Topic modeling with BERTopic

Discovers themes across reviews without predefined categories.
Uses the dense embeddings from embeddings.py so we don't re-encode.

BERTopic pipeline:
1. Embeddings (already computed) — dense vector per review
2. UMAP — reduce 384 dimensions to ~5 for clustering
3. HDBSCAN — find dense clusters of similar reviews
4. c-TF-IDF — extract top words per cluster to label topics

This is unsupervised — you don't tell it what topics to find,
it discovers them from the data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bertopic import BERTopic

from preprocessing import load_cleaned_data

Path("figures").mkdir(exist_ok=True)


# topic modeling

def build_topic_model(texts, embeddings, min_topic_size=50):
    """
    Build a BERTopic model from texts and precomputed embeddings.

    min_topic_size controls how many reviews a cluster needs to count
    as a topic. Lower = more topics (finer-grained), higher = fewer
    topics (broader themes). 50 is a reasonable default for 50k reviews.

    Reviews that don't fit any cluster get assigned topic -1 (outliers).
    These are worth looking at — they might be unusual reviews or just
    ones that don't fit neatly into any theme.
    """
    print("Building BERTopic model...")

    # BERTopic uses its own UMAP and HDBSCAN internally, but since we're
    # passing precomputed embeddings, it skips the embedding step.
    model = BERTopic(
        min_topic_size=min_topic_size,
        verbose=True,
    )

    topics, probs = model.fit_transform(texts, embeddings=embeddings)

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = sum(1 for t in topics if t == -1)

    print(f"\nFound {n_topics} topics")
    print(f"Outlier reviews (topic -1): {n_outliers:,} ({n_outliers/len(texts)*100:.1f}%)")

    return model, topics, probs


def get_topic_summary(model, top_n=10):
    """Get top words for each topic as a readable summary."""
    topic_info = model.get_topic_info()
    # filter out outlier topic (-1)
    topic_info = topic_info[topic_info["Topic"] != -1].head(top_n)

    summaries = []
    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]
        count = row["Count"]
        words = model.get_topic(topic_id)
        top_words = ", ".join([w for w, _ in words[:5]])
        summaries.append({
            "topic": topic_id,
            "count": count,
            "top_words": top_words,
        })

    return pd.DataFrame(summaries)


# analysis

def topic_sentiment_breakdown(df, topics):
    """
    For each topic, what percentage of reviews are positive vs negative?

    Some topics might be mostly complaints (negative), others mostly praise.
    This helps identify which product aspects drive satisfaction vs frustration.
    """
    df = df.copy()
    df["topic"] = topics

    # exclude outliers
    topic_df = df[df["topic"] != -1]

    breakdown = (
        topic_df.groupby("topic")["rating"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "pct_positive", "count": "n_reviews"})
        .sort_values("n_reviews", ascending=False)
    )
    breakdown["pct_positive"] = (breakdown["pct_positive"] * 100).round(1)

    return breakdown


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_topic_bars(summary_df, save_path="figures/topic_distribution.png"):
    """Plot topic sizes as a bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [f"Topic {row['topic']}: {row['top_words'][:30]}..."
              for _, row in summary_df.iterrows()]
    counts = summary_df["count"].values

    ax.barh(range(len(labels)), counts, color="#3498db", edgecolor="black", alpha=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Number of Reviews")
    ax.set_title("Top Topics by Size")
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_topic_sentiment(breakdown, model, save_path="figures/topic_sentiment.png"):
    """Plot sentiment distribution per topic."""
    top = breakdown.head(10)

    fig, ax = plt.subplots(figsize=(12, 6))

    topic_labels = []
    for topic_id in top.index:
        words = model.get_topic(topic_id)
        label = f"T{topic_id}: {', '.join([w for w, _ in words[:3]])}"
        topic_labels.append(label)

    x = range(len(top))
    colors = ["#2ecc71" if p > 60 else "#e74c3c" if p < 40 else "#f39c12"
              for p in top["pct_positive"]]

    ax.bar(x, top["pct_positive"], color=colors, edgecolor="black", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(topic_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("% Positive Reviews")
    ax.set_title("Sentiment by Topic (Green >60%, Red <40%)")
    ax.axhline(50, color="black", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":

    df = load_cleaned_data()
    if df is None:
        raise RuntimeError("Run preprocessing.py first")

    # load precomputed embeddings from embeddings.py
    emb_path = Path("data/dense_embeddings.npy")
    if not emb_path.exists():
        raise RuntimeError("Run embeddings.py first to generate dense embeddings")

    embeddings = np.load(emb_path)
    texts = df["clean_text"].tolist()

    # building topics

    model, topics, probs = build_topic_model(texts, embeddings)

    # summarize topics

    summary = get_topic_summary(model, top_n=15)
    print("\n=== TOP TOPICS ===")
    for _, row in summary.iterrows():
        print(f"  Topic {row['topic']:3d} ({row['count']:,} reviews): {row['top_words']}")

    # sentiment per topic

    breakdown = topic_sentiment_breakdown(df, topics)
    print("\n=== TOPIC SENTIMENT ===")
    for topic_id, row in breakdown.head(10).iterrows():
        words = model.get_topic(topic_id)
        label = ", ".join([w for w, _ in words[:3]])
        print(f"  Topic {topic_id:3d}: {row['pct_positive']:.0f}% positive "
              f"({row['n_reviews']:,} reviews) — {label}")

    # outlier analysis
    # Reviews assigned to topic -1 didn't fit any cluster.
    # Often these are unusual, very short, or very niche reviews.

    outlier_idx = [i for i, t in enumerate(topics) if t == -1]
    print(f"\n=== OUTLIER REVIEWS ({len(outlier_idx):,}) ===")
    print("Sample outliers:")
    for i in outlier_idx[:5]:
        print(f"  [{df['rating'].iloc[i]}] {texts[i][:120]}...")

    
    # save topic assignments for use by rag.py
    df_out = df.copy()
    df_out["topic"] = topics
    df_out.to_parquet("data/reviews_with_topics.parquet")

    print("\n\nDone. Topics saved to data/reviews_with_topics.parquet")
