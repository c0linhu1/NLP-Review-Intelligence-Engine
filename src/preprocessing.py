"""
preprocessing.py — Text cleaning and tokenization pipeline

cleaning raw test- HTML tags, weird unicode, inconsistent casing, extra whitespace
"""

import re
import html
import unicodedata
from pathlib import Path

import pandas as pd
import spacy
from datasets import load_dataset


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
        .sample(n=n_samples, random_state = random_state)
        .reset_index(drop = True)  
    )

    df.rename(columns={"content": "text", "label": "rating"}, inplace = True)

    print(f"Label distribution:\n{df['rating'].value_counts().sort_index().to_string()}")

    # clean both the review body and the title 
    df["clean_text"] = df["text"].apply(clean_text)
    df["clean_title"] = df["title"].apply(clean_text)

    # Some reviews might become empty strings after cleaning
    # (e.g., reviews that were only HTML tags or non-ASCII characters).
    # Drop these since they'll cause errors downstream.
    before = len(df)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    if before - len(df) > 0:
        print(f"Dropped {before - len(df)} empty reviews after cleaning")

    if save:
        Path("data").mkdir(exist_ok=True)
        df.to_parquet(cache_path)
        print(f"Saved cleaned data to {cache_path}")

    return df


def clean_text(text):
    """
    Clean a single review string.

    1. HTML unescape — Amazon reviews sometimes contain HTML entities like
       &amp; (should be &) or &lt; (should be <). html.unescape() converts
       these back to normal characters.

    2. Strip HTML tags — Some reviews contain actual HTML like <br/> for line
       breaks or <b> for bold. We replace these with a space (not nothing,
       because "word<br/>word" should become "word word" not "wordword").

    3. Unicode normalize — Characters like é, ñ, ü get decomposed into their
       base letter + accent mark, then we strip the accent. This prevents
       "café" and "cafe" from being treated as different tokens.
       NFKD = Normalization Form Compatibility Decomposition.

    4. Lowercase — "Great", "GREAT", and "great" should all be the same token.
       Almost every NLP pipeline lowercases unless you specifically need case
       information (like NER where "Apple" the company ≠ "apple" the fruit).

    5. Collapse whitespace — After the previous steps, we might have double
       spaces, tabs, or leftover newlines. Normalize everything to single spaces.
    """
    # handle None values or non-string inputs gracefully.
    if not isinstance(text, str) or len(text) == 0:
        return ""

    # Step 1: HTML entities → normal characters
    # "This product is &lt;amazing&gt;" → "This product is <amazing>"
    text = html.unescape(text)

    # Step 2: Remove HTML tags - using regex for this
    # "Great product<br/>Would buy again" → "Great product Would buy again"
    text = re.sub(r"<[^>]+>", " ", text)

    # Step 3: Normalize unicode and strip non-ASCII
    # "café" → "cafe", "naïve" → "naive"
    # encode to ASCII (dropping anything that can't be represented) then decode back.
    # it drops things like Chinese characters or emojis.
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Step 4: Lowercase everything
    # "I LOVE this product" → "i love this product"
    text = text.lower()

    # Step 5: Collapse whitespace
    # "  hello   world  \n\t test  " → "hello world test"
    text = re.sub(r"\s+", " ", text).strip()

    return text



# Tokenization

# spaCy handles edge cases the best compared to regex or whitespace
_nlp = None

def _get_nlp():
    """
    Lazy-load spaCy model
    """
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def tokenize_text(text):
    """
    Tokenize a single cleaned text string using spaCy.

    Use this for one-off tokenization (e.g., in the Streamlit app when a user
    types a query). For batch tokenization of the full dataset, use
    tokenize_corpus() instead — it's much faster.
    """
    nlp = _get_nlp()
    # tokenize one string
    doc = nlp(text)

    # Filter out whitespace-only tokens (spaces, tabs, newlines).
    # spaCy creates tokens for these but they carry no meaning.
    return [token.text for token in doc if not token.is_space]


def tokenize_corpus(df, text_col = "clean_text", batch_size = 500):
    """
    Tokenize an entire DataFrame column using spaCy's pipe for speed.

    WHY nlp.pipe() INSTEAD OF .apply()?
    spaCy's pipe() processes documents in batches internally, which is
    significantly faster than calling nlp() on each string individually.
    On 50k reviews, this is the difference between 2 minutes and 10+ minutes.

    WHY DISABLE NER AND PARSER?
    spaCy's pipeline has multiple components: tokenizer, tagger, parser, NER.
    We only need the tokenizer right now. Disabling the rest makes it ~3x faster.
    When we need NER (aspect extraction) or dependency parsing later, we'll
    run spaCy again with those components enabled.

    Adds columns:
        - tokens: list of token strings per review
        - n_tokens: token count per review (useful for EDA and feature engineering)
    """
    nlp = _get_nlp()

    print(f"Tokenizing {len(df):,} reviews (this takes a few minutes)...")

    tokenized = []
    for doc in nlp.pipe(df[text_col].tolist(), batch_size = batch_size,
                        disable = ["ner", "parser"]):
        tokens = [token.text for token in doc if not token.is_space]
        tokenized.append(tokens)

    df = df.copy()
    df["tokens"] = tokenized
    df["n_tokens"] = df["tokens"].apply(len)

    print(f"Done. Avg tokens/review: {df['n_tokens'].mean():.1f}, "
          f"total tokens: {df['n_tokens'].sum():,}")

    return df



# utility 
def get_content_tokens(tokens, stop_words = None, min_length = 2):
    """
    Filter a token list to content words only (no stopwords, no short tokens).



    we dont remove stopwords during the main preprocessing step because
    some components need them. For example:
        - The sentiment classifier benefits from negation words ("not", "never")
        - The summarizer needs full sentences with proper grammar
        - Dependency parsing needs the full sentence structure

    Args:
        tokens: list of token strings
        stop_words: set of words to remove (defaults to spaCy's English stopwords)
        min_length: minimum character length to keep (filters out "a", "I", etc.)
    """
    if stop_words is None:
        from spacy.lang.en.stop_words import STOP_WORDS
        stop_words = STOP_WORDS
    return [t for t in tokens if t not in stop_words and len(t) >= min_length]



if __name__ == "__main__":

    df = load_and_clean_data()
    df = tokenize_corpus(df)

    # Save the tokenized version (overwrites the pre-tokenization parquet)
    df.to_parquet("data/reviews_clean.parquet")

    # sanity-check the output
    print(f"\nSample cleaned reviews:")
    for i in range(3):
        print(f"\n--- Review {i} ---")
        print(f"Rating: {'positive' if df['rating'].iloc[i] == 1 else 'negative'}")
        print(f"Title:  {df['clean_title'].iloc[i]}")
        print(f"Text:   {df['clean_text'].iloc[i][:200]}")
        print(f"Tokens: {df['tokens'].iloc[i][:20]}...")