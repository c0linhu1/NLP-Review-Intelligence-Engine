"""
embeddings.py — Sparse & dense representations + FAISS similarity search

Converts reviews into vectors two ways:
1. TF-IDF (sparse) — bag-of-words weighted by word importance
2. Sentence-transformers (dense) — contextual embeddings from a pretrained model

Then builds a FAISS index for fast similarity search on the dense vectors.
The FAISS index gets reused by the RAG engine and topic modeler.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import umap
from preprocessing import load_and_clean_data

# sparse embeddings

def build_tfidf(texts, max_features = 5000):
    """
    based off of bag of words - word counts are reweighted based on how informative they 
    are across corpus
    """
    vectorizer = TfidfVectorizer(max_features = max_features)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def tfidf_search(query, vectorizer, tfidf_matrix, texts, top_k=5):
    """
    Search for similar reviews using TF-IDF cosine similarity.

    Transforms the query into the same TF-IDF space, then finds the
    reviews with the highest cosine similarity. Simple but limited —
    only matches on exact word overlap not similar words like cozy and comfortable
    """
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = scores.argsort()[-top_k:][::-1]
    return [(texts[i], scores[i]) for i in top_idx]



# dense embeddings (sentence transformers)

def build_dense_embeddings(texts, model_name="all-MiniLM-L6-v2", batch_size=256,
                           save_path="data/dense_embeddings.npy"):
    """
    Encode texts into dense vectors using a sentence-transformer model.

    Unlike TF-IDF, this understands meaning — "cozy" and "comfortable"
    end up close together in the embedding space because the model learned
    from millions of text pairs that they're used in similar contexts.

    all-MiniLM-L6-v2 is small (80MB) and fast while still being accurate.
    It outputs 384-dimensional vectors for each input text.

    We save the embeddings to disk so we don't have to re-encode 50k
    reviews every time we run a downstream script.
    """
    cache = Path(save_path)
    if cache.exists():
        print(f"Loading cached embeddings from {cache}")
        return np.load(cache)

    print(f"Encoding {len(texts):,} texts with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    # float 32 bc FAISS requires float 32 - pytorch default is float 64
    embeddings = np.array(embeddings).astype("float32")

    Path(save_path).parent.mkdir(exist_ok=True)
    np.save(cache, embeddings)
    print(f"Saved embeddings to {cache} — shape: {embeddings.shape}")

    return embeddings


def get_sentence_model(model_name="all-MiniLM-L6-v2"):
    """Load sentence-transformer model for encoding new queries."""
    return SentenceTransformer(model_name)


# FAISS index

def build_faiss_index(embeddings, save_path="data/faiss_index.bin"):
    """
    Build a FAISS index for fast nearest-neighbor search.

    Why FAISS instead of just cosine_similarity on the full matrix?
    cosine_similarity compares your query against every single vector —
    fine for 1000 reviews, painfully slow for 1M+. FAISS uses optimized
    data structures to find approximate nearest neighbors much faster.

    IndexFlatIP = exact inner product search (no approximation).
    For 50k vectors this is fast enough. For millions you'd use
    IndexIVFFlat or IndexHNSW for approximate but faster search.

    We normalize vectors first so inner product = cosine similarity.
    """
    cache = Path(save_path)
    if cache.exists():
        print(f"Loading cached FAISS index from {cache}")
        return faiss.read_index(str(cache))

    # normalize so inner product = cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms

    dim = normalized.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(normalized)

    faiss.write_index(index, str(cache))
    print(f"Built FAISS index: {index.ntotal} vectors, {dim} dimensions")

    return index


def faiss_search(query_text, model, index, texts, top_k=5):
    """
    Search the FAISS index with a natural language query.

    Encode the query → normalize → search → return top-k results.
    """
    query_vec = model.encode([query_text]).astype("float32")
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

    scores, indices = index.search(query_vec, top_k)
    return [(texts[i], scores[0][j]) for j, i in enumerate(indices[0])]




if __name__ == "__main__":

    Path("figures").mkdir(exist_ok=True)

    # load cleaned data
    df = load_and_clean_data()

    texts = df["clean_text"].tolist()

    # ------------------------------------------------------------------
    # 1. BUILD BOTH REPRESENTATIONS
    # ------------------------------------------------------------------

    print("=== BUILDING TF-IDF ===")
    vectorizer, tfidf_matrix = build_tfidf(texts)
    print(f"TF-IDF matrix: {tfidf_matrix.shape}")

    print("\n=== BUILDING DENSE EMBEDDINGS ===")
    embeddings = build_dense_embeddings(texts)
    print(f"Dense embeddings: {embeddings.shape}")

    print("\n=== BUILDING FAISS INDEX ===")
    index = build_faiss_index(embeddings)

    # save the vectorizer for later use by classifier
    with open("data/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print("Saved TF-IDF vectorizer to data/tfidf_vectorizer.pkl")

    # ------------------------------------------------------------------
    # 2. COMPARE SPARSE VS DENSE SEARCH
    # ------------------------------------------------------------------
    # This is the key insight: TF-IDF only matches on exact word overlap.
    # Dense embeddings understand meaning — "cozy" matches "comfortable"
    # even though they share zero characters.

    model = get_sentence_model()

    queries = [
        "comfortable chair for reading",
        "broke after two weeks",
        "great gift for my daughter",
        "terrible customer service experience",
    ]

    print("\n\n=== SPARSE VS DENSE SEARCH COMPARISON ===")
    for query in queries:
        print(f"\nQuery: '{query}'")

        print("\n  TF-IDF results:")
        tfidf_results = tfidf_search(query, vectorizer, tfidf_matrix, texts, top_k=3)
        for text, score in tfidf_results:
            print(f"    [{score:.3f}] {text[:100]}...")

        print("\n  Dense results:")
        dense_results = faiss_search(query, model, index, texts, top_k=3)
        for text, score in dense_results:
            print(f"    [{score:.3f}] {text[:100]}...")

