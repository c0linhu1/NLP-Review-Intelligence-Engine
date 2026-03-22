"""
preprocessing.py — Text cleaning and tokenization pipeline

cleaning raw text - HTML tags, weird unicode, inconsistent casing, extra whitespace
also includes corpus exploration: Zipf's law, stopwords, TF-IDF, n-grams, EDA
"""

import re
import html
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from datasets import load_dataset
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer



def load_and_clean_data(n_samples=50000, random_state=42, save=True):
    """
    Load Amazon Polarity dataset, sample, clean, and optionally save to parquet.

    Parquet is a columnar file format - faster to read/write than CSV,
    preserves data types (CSVs lose list columns), and compresses well.
    """

    # Check if we already have cleaned data saved — avoid re-downloading
    cache_path = Path("data/reviews_clean.parquet")
    if cache_path.exists():
        print(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    # downloads the dataset the first time then caches it locally
    print("Loading Amazon Polarity dataset...")
    dataset = load_dataset("amazon_polarity", split="train")

    # convert HuggingFace Dataset → pandas DataFrame and sample down
    df = (
        pd.DataFrame(dataset)
        .sample(n=n_samples, random_state=random_state)
        .reset_index(drop=True)
    )

    df.rename(columns={"content": "text", "label": "rating"}, inplace=True)

    print(f"Label distribution:\n{df['rating'].value_counts().sort_index().to_string()}")

    # clean both the review body and the title
    df["clean_text"] = df["text"].apply(clean_text)
    df["clean_title"] = df["title"].apply(clean_text)

    # Some reviews might become empty strings after cleaning
    before = len(df)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    if before - len(df) > 0:
        print(f"Dropped {before - len(df)} empty reviews after cleaning")

    if save:
        Path("data").mkdir(exist_ok=True)
        df.to_parquet(cache_path)
        print(f"Saved cleaned data to {cache_path}")

    return df



# cleaning
def clean_text(text):
    """
    Clean a single review string.

    1. HTML unescape — &amp; → &, &lt; → <
    2. Strip HTML tags — <br/>, <b> → space
    3. Unicode normalize — é → e, ñ → n
    4. Lowercase — "Great" and "GREAT" → "great"
    5. Collapse whitespace — tabs, newlines, double spaces → single space
    """
    if not isinstance(text, str) or len(text) == 0:
        return ""

    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()

    return text


# tokenization
_nlp = None

def _get_nlp():
    """Lazy-load spaCy model"""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def tokenize_text(text):
    """Tokenize a single cleaned text string using spaCy."""
    nlp = _get_nlp()
    doc = nlp(text)
    return [token.text for token in doc if not token.is_space]


def tokenize_corpus(df, text_col="clean_text", batch_size=500):
    """
    Tokenize an entire DataFrame column using spaCy's pipe for speed.

    nlp.pipe() processes in batches — much faster than calling nlp() individually.
    We disable NER and parser since we only need the tokenizer here (~3x faster).
    """
    nlp = _get_nlp()

    print(f"Tokenizing {len(df):,} reviews (this takes a few minutes)...")

    tokenized = []
    for doc in nlp.pipe(df[text_col].tolist(), batch_size=batch_size,
                        disable=["ner", "parser"]):
        tokens = [token.text for token in doc if not token.is_space]
        tokenized.append(tokens)

    df = df.copy()
    df["tokens"] = tokenized
    df["n_tokens"] = df["tokens"].apply(len)

    print(f"Done. Avg tokens/review: {df['n_tokens'].mean():.1f}, "
          f"total tokens: {df['n_tokens'].sum():,}")

    return df

# utility
def get_content_tokens(tokens, stop_words=None, min_length=2):
    """
    Filter a token list to content words only (no stopwords, no short tokens).

    We dont remove stopwords during main preprocessing because some components
    need them — sentiment classifier needs "not", dependency parsing needs
    full sentence structure. Filter here only when needed.
    """
    if stop_words is None:
        stop_words = STOP_WORDS
    return [t for t in tokens if t not in stop_words and len(t) >= min_length]


def get_ngrams(tokens, n):
    """
    extract n-grams (sequences of n consecutive tokens) from a token list.
    bigrams (n=2): ["I", "love", "this"] → [("I", "love"), ("love", "this")]
    """
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]




if __name__ == "__main__":

    Path("figures").mkdir(exist_ok = True)

    df = load_and_clean_data()
    df = tokenize_corpus(df)
    df.to_parquet("data/reviews_clean.parquet")

    print(f"\nSample cleaned reviews:")
    for i in range(3):
        print(f"\n--- Review {i} ---")
        print(f"Rating: {'positive' if df['rating'].iloc[i] == 1 else 'negative'}")
        print(f"Title:  {df['clean_title'].iloc[i]}")
        print(f"Text:   {df['clean_text'].iloc[i][:200]}")
        print(f"Tokens: {df['tokens'].iloc[i][:20]}...")

    # tokenization comparison
    # Three methods on the same sentence to see why spaCy wins.
    # Key difference: how each handles "don't" and "isn't".
    # Negation matters — if your tokenizer destroys it,
    # "isn't great" becomes indistinguishable from "great".

    nlp = _get_nlp()
    sample = "I don't think this $30 coffee-maker is worth it... the build quality isn't great!"

    print("\n\n=== TOKENIZATION COMPARISON ===")
    print(f"Input: {sample}\n")
    print("WHITESPACE:", sample.lower().split())
    print("REGEX:     ", re.findall(r'\b\w+\b', sample.lower()))
    print("SPACY:     ", [t.text for t in nlp(sample.lower())])


    # Zipf's law: word frequency is inversely proportional to rank.
    # A tiny number of words dominate the corpus, most words are rare.

    all_tokens = [t for tokens in df["tokens"] for t in tokens]
    vocab = Counter(all_tokens)

    print(f"\n\n=== VOCABULARY ===")
    print(f"Total tokens: {len(all_tokens):,}")
    print(f"Unique tokens: {len(vocab):,}")
    print(f"\nTop 20 most common:")
    for word, count in vocab.most_common(20):
        print(f"  {word:15s} {count:>7,}")

    ranks = np.arange(1, len(vocab) + 1)
    frequencies = sorted(vocab.values(), reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(ranks[:500], frequencies[:500])
    axes[0].set_xlabel("Rank")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Word Frequency vs Rank (Top 500)")
    axes[1].loglog(ranks, frequencies)
    axes[1].set_xlabel("Rank (log)")
    axes[1].set_ylabel("Frequency (log)")
    axes[1].set_title("Zipf's Law (Log-Log)")
    plt.tight_layout()
    plt.savefig("figures/zipfs_law.png", dpi=150, bbox_inches="tight")
    plt.show()

    
    # STOPWORD ANALYSIS

    # Removing stopwords cuts ~40% of tokens and focuses on content.
    # But be careful — negation words like "not" matter for sentiment.

    content_tokens = [t for t in all_tokens if t not in STOP_WORDS and len(t) > 1]
    content_vocab = Counter(content_tokens)
    pct_removed = (len(all_tokens) - len(content_tokens)) / len(all_tokens) * 100

    print(f"\n\n=== STOPWORD ANALYSIS ===")
    print(f"With stopwords:    {len(all_tokens):>10,} tokens, {len(vocab):>7,} unique")
    print(f"Without stopwords: {len(content_tokens):>10,} tokens, {len(content_vocab):>7,} unique")
    print(f"Removed: {pct_removed:.1f}% of tokens")
    print(f"\nTop 20 content words:")
    for word, count in content_vocab.most_common(20):
        print(f"  {word:15s} {count:>7,}")


    # TF-IDF = Term Frequency × Inverse Document Frequency
    # Words that are frequent in a doc but rare across all docs get high scores.
    # "the" → appears everywhere → low score. "ergonomic" → rare → high score.

    vectorizer = TfidfVectorizer(max_features = 5000)
    tfidf_matrix = vectorizer.fit_transform(df["clean_text"].head(1000))
    feature_names = vectorizer.get_feature_names_out()

    print(f"\n\n=== TF-IDF ===")
    print(f"Matrix shape: {tfidf_matrix.shape} (docs × features)")

    row = tfidf_matrix[0].toarray().flatten()
    top_indices = row.argsort()[-10:][::-1]

    print(f"\nReview: {df['clean_text'].iloc[0][:200]}")
    print(f"\nTop TF-IDF terms for this review:")
    for idx in top_indices:
        print(f"  {feature_names[idx]:15s} TF-IDF={row[idx]:.4f}")

    # ------------------------------------------------------------------
    # 6. REVIEW LENGTH & SENTIMENT EDA
    # ------------------------------------------------------------------

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].hist(df["n_tokens"], bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_xlabel("Number of Tokens")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Review Length Distribution")
    axes[0].axvline(df["n_tokens"].median(), color="red", linestyle="--",
                    label=f"Median: {df['n_tokens'].median():.0f}")
    axes[0].legend()

    df.boxplot(column="n_tokens", by="rating", ax=axes[1])
    axes[1].set_xlabel("Sentiment (0=neg, 1=pos)")
    axes[1].set_ylabel("Number of Tokens")
    axes[1].set_title("Review Length by Sentiment")
    plt.sca(axes[1])
    plt.title("Review Length by Sentiment")

    df["rating"].value_counts().sort_index().plot(
        kind="bar", ax=axes[2], edgecolor="black", alpha=0.7,
        color=["#e74c3c", "#2ecc71"]
    )
    axes[2].set_xlabel("Sentiment")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Label Distribution")
    axes[2].set_xticklabels(["Negative", "Positive"], rotation=0)

    plt.tight_layout()
    plt.savefig("figures/review_eda.png", dpi=150, bbox_inches="tight")
    plt.show()

    for label, name in [(0, "Negative"), (1, "Positive")]:
        subset = df[df["rating"] == label]
        print(f"{name}: avg {subset['n_tokens'].mean():.1f} tokens, "
              f"median {subset['n_tokens'].median():.0f} tokens")

    # ------------------------------------------------------------------
    # 7. N-GRAM ANALYSIS
    # ------------------------------------------------------------------
    # Single words miss multi-word patterns. "not good" as unigrams
    # looks like it contains "good" (positive). As a bigram it's
    # clearly negative.

    all_content = [get_content_tokens(tokens) for tokens in df["tokens"]]

    bigrams = Counter()
    trigrams = Counter()
    for tokens in all_content:
        bigrams.update(get_ngrams(tokens, 2))
        trigrams.update(get_ngrams(tokens, 3))

    print(f"\n\n=== N-GRAM ANALYSIS ===")
    print("Top 15 Bigrams:")
    for gram, count in bigrams.most_common(15):
        print(f"  {' '.join(gram):25s} {count:>6,}")

    print("\nTop 15 Trigrams:")
    for gram, count in trigrams.most_common(15):
        print(f"  {' '.join(gram):30s} {count:>6,}")

    # compare bigrams across sentiment labels
    pos_bigrams = Counter()
    neg_bigrams = Counter()
    for i, tokens in enumerate(all_content):
        grams = get_ngrams(tokens, 2)
        if df["rating"].iloc[i] == 1:
            pos_bigrams.update(grams)
        else:
            neg_bigrams.update(grams)

    print("\nMostly-Positive Bigrams:")
    count = 0
    for gram, freq in pos_bigrams.most_common(50):
        neg_freq = neg_bigrams.get(gram, 0)
        if neg_freq < freq * 0.3 and freq > 50:
            print(f"  {' '.join(gram):25s} pos={freq:>5,}  neg={neg_freq:>5,}")
            count += 1
            if count >= 10:
                break

    print("\nMostly-Negative Bigrams:")
    count = 0
    for gram, freq in neg_bigrams.most_common(50):
        pos_freq = pos_bigrams.get(gram, 0)
        if pos_freq < freq * 0.3 and freq > 50:
            print(f"  {' '.join(gram):25s} neg={freq:>5,}  pos={pos_freq:>5,}")
            count += 1
            if count >= 10:
                break
