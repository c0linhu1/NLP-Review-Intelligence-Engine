"""
rag.py — Retrieval Augmented Generation + Prompt Engineering

RAG is the most in-demand NLP pattern in production right now. The idea:
1. User asks a question
2. Retrieve relevant documents (reviews) from FAISS
3. Inject those documents into a prompt
4. LLM generates an answer grounded in the retrieved documents

Without RAG, the LLM hallucinates — it makes up answers that sound right
but aren't based on any real data. With RAG, it can only use information
from the retrieved reviews.

This file also covers prompt engineering: system prompts, few-shot examples,
temperature comparison, and controlling output format.
"""

import os
from pathlib import Path

import numpy as np
import faiss
import anthropic
from sentence_transformers import SentenceTransformer

from preprocessing import load_cleaned_data

Path("figures").mkdir(exist_ok=True)


# retrieva;

def load_retrieval_components():
    """Load the FAISS index and sentence model for retrieval."""
    index = faiss.read_index("data/faiss_index.bin")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, model


def retrieve_reviews(query, model, index, texts, top_k=5):
    """
    Retrieve the top-k most relevant reviews for a query.

    This is the "R" in RAG — retrieval. We encode the query into the
    same embedding space as the reviews, then find the nearest neighbors.
    """
    query_vec = model.encode([query]).astype("float32")
    query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

    scores, indices = index.search(query_vec, top_k)

    results = []
    for j, i in enumerate(indices[0]):
        results.append({
            "text": texts[i],
            "score": float(scores[0][j]),
            "index": int(i),
        })

    return results

# prompt engineering

def build_rag_prompt(question, retrieved_reviews):
    """
    Build a prompt that grounds the LLM's answer in retrieved reviews.

    This is the "A" in RAG — augmentation. We inject the retrieved reviews
    into the prompt so the model has real data to reference. The system
    prompt instructs it to ONLY use information from the reviews.

    Key prompt engineering decisions:
    - Number each review so the model can cite sources
    - Explicitly tell it to say "I don't know" if reviews don't cover the question
    - Ask it to cite which review numbers support each claim
    """
    context = ""
    for i, review in enumerate(retrieved_reviews, 1):
        context += f"[Review {i}] (similarity: {review['score']:.3f})\n"
        context += f"{review['text'][:500]}\n\n"

    prompt = f"""Based on the following customer reviews, answer the user's question.

RULES:
- Only use information from the reviews below. Do not make anything up.
- Cite which review number(s) support each claim, like [Review 1].
- If the reviews don't address the question, say so honestly.
- Be specific — use details from the reviews, not generic statements.

REVIEWS:
{context}

QUESTION: {question}"""

    return prompt


def build_summary_prompt(reviews, style="paragraph"):
    """
    Build a prompt for summarizing a batch of reviews.

    Demonstrates prompt engineering concepts:
    - style parameter: control output format through instructions
    - few-shot example: show the model what a good summary looks like
    - grounding instruction: only use information from the reviews
    """
    review_text = ""
    for i, review in enumerate(reviews, 1):
        review_text += f"[{i}] {review[:300]}\n"

    if style == "paragraph":
        format_instruction = "Write a concise paragraph summarizing the overall sentiment and key themes."
    elif style == "bullets":
        format_instruction = "Summarize as bullet points: one bullet per key theme."
    elif style == "structured":
        format_instruction = """Summarize in this format:
OVERALL: one sentence overall sentiment
PROS: bullet points of positive themes
CONS: bullet points of negative themes
VERDICT: one sentence takeaway"""
    else:
        format_instruction = "Write a concise summary."

    prompt = f"""Summarize the following customer reviews.

{format_instruction}

Only reference information that appears in the reviews. Be specific.

REVIEWS:
{review_text}"""

    return prompt

# llm calls

def call_llm(prompt, system_prompt=None, temperature=0.3, max_tokens=500):
    """
    Call the Anthropic API.

    temperature controls randomness:
    - 0.0 = deterministic, always picks the most likely token
    - 0.3 = slight variation, good for factual tasks like RAG
    - 1.0 = more creative, good for brainstorming, bad for factual answers

    For RAG you want low temperature because you want accurate, grounded answers.
    For creative summarization you might go higher.
    """
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

    messages = [{"role": "user", "content": prompt}]

    kwargs = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": max_tokens,
        "messages": messages,
        "temperature": temperature,
    }

    if system_prompt:
        kwargs["system"] = system_prompt

    response = client.messages.create(**kwargs)
    return response.content[0].text


def rag_answer(question, model, index, texts, temperature=0.3):
    """
    Full RAG pipeline: retrieve → build prompt → generate answer.
    Returns the answer and the retrieved reviews for source attribution.
    """
    # retrieve
    retrieved = retrieve_reviews(question, model, index, texts, top_k=5)

    # augment
    prompt = build_rag_prompt(question, retrieved)

    system = ("You are a helpful assistant that answers questions about products "
              "based on customer reviews. Be accurate, specific, and cite your sources.")

    # generate
    answer = call_llm(prompt, system_prompt=system, temperature=temperature)

    return answer, retrieved


def summarize_reviews(reviews, style="paragraph", temperature=0.3):
    """Generate a summary of a batch of reviews."""
    prompt = build_summary_prompt(reviews, style=style)

    system = ("You are a product analyst summarizing customer feedback. "
              "Be concise and accurate. Only state what the reviews actually say.")

    return call_llm(prompt, system_prompt=system, temperature=temperature)



if __name__ == "__main__":

    # check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set your API key: export ANTHROPIC_API_KEY='your-key-here'")
        print("Get one at: https://console.anthropic.com/")
        exit(1)

    df = load_cleaned_data()
    if df is None:
        raise RuntimeError("Run preprocessing.py first")

    texts = df["clean_text"].tolist()

    # load retrieval components from embeddings.py output
    index, sent_model = load_retrieval_components()

    # rag q&a
    # Ask questions and get answers grounded in real reviews.

    questions = [
        "Are these products easy to assemble?",
        "What do customers say about durability?",
        "Are customers happy with the value for money?",
        "What are the most common complaints?",
    ]

    print("=== RAG Q&A ===\n")
    for question in questions:
        print(f"Q: {question}")
        answer, sources = rag_answer(question, sent_model, index, texts)
        print(f"A: {answer}\n")
        print(f"Sources ({len(sources)} reviews retrieved):")
        for s in sources[:3]:
            print(f"  [{s['score']:.3f}] {s['text'][:100]}...")
        print("\n" + "=" * 60 + "\n")

    # rag vs no retrieval
    # Show why RAG matters: without retrieval context, the model hallucinates.

    print("=== RAG VS NO RETRIEVAL ===\n")
    test_q = "What do customers think about the quality of kitchen products?"

    # with RAG
    rag_ans, _ = rag_answer(test_q, sent_model, index, texts)

    # without RAG — just ask the question directly with no review context
    no_rag_ans = call_llm(
        f"Based on Amazon product reviews, {test_q}",
        temperature=0.3
    )

    print(f"Q: {test_q}\n")
    print(f"WITH RAG:\n{rag_ans}\n")
    print(f"WITHOUT RAG (no retrieval):\n{no_rag_ans}\n")
    print("Notice: without RAG, the model gives generic/made-up answers.")
    print("With RAG, it cites specific reviews.\n")

    # prompt engineering
    # Same reviews, different prompt instructions → different output formats.

    print("=== SUMMARY STYLES ===\n")
    sample_reviews = texts[:20]  # grab first 20 reviews

    for style in ["paragraph", "bullets", "structured"]:
        print(f"--- Style: {style} ---")
        summary = summarize_reviews(sample_reviews, style=style)
        print(summary)
        print()

    # temperature comparison
    # Same prompt at different temperatures to see how randomness affects output.
    # Low temp = consistent and factual. High temp = more varied and creative.

    print("=== TEMPERATURE COMPARISON ===\n")
    for temp in [0.0, 0.5, 1.0]:
        print(f"--- Temperature: {temp} ---")
        summary = summarize_reviews(sample_reviews[:10], style="paragraph",
                                     temperature=temp)
        print(summary)
        print()

    # ------------------------------------------------------------------
    # 5. FEW-SHOT PROMPTING
    # ------------------------------------------------------------------
    # Include an example of what a good answer looks like in the prompt.
    # This steers the model's output format and quality.

    print("=== FEW-SHOT PROMPTING ===\n")

    few_shot_prompt = """Answer customer questions based on reviews. Here's an example:

EXAMPLE REVIEWS:
[1] Great blender, very powerful. Crushed ice easily.
[2] The motor died after 3 months. Very disappointed.
[3] Love this blender! Makes perfect smoothies every morning.

EXAMPLE QUESTION: Is this blender reliable?
EXAMPLE ANSWER: Customer opinions are mixed on reliability. Most reviewers praise its power and performance, noting it crushes ice easily [Review 1] and makes great smoothies [Review 3]. However, at least one customer reported the motor failing after just 3 months [Review 2], raising durability concerns.

Now answer this question the same way:

REVIEWS:
"""
    # add real retrieved reviews
    test_q2 = "Would customers recommend these products?"
    retrieved = retrieve_reviews(test_q2, sent_model, index, texts, top_k=5)

    for i, r in enumerate(retrieved, 1):
        few_shot_prompt += f"[{i}] {r['text'][:300]}\n"

    few_shot_prompt += f"\nQUESTION: {test_q2}"

    answer = call_llm(few_shot_prompt, temperature=0.3)
    print(f"Q: {test_q2}")
    print(f"A: {answer}")

    print("\n\nDone.")