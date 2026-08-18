import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# LLaMA-3.1 8B on Groq — free, 800+ tokens/sec, fast enough for live demo
MODEL = "openai/gpt-oss-20b"

# temperature 0.1 = very factual, low creativity
# don't want it making things up
TEMP = 0.1


SYSTEM_PROMPT = """You are AskMyDocs, a document Q&A assistant.

Rules:
1. Only answer from the provided context. If the answer isn't there, say so.
2. Always mention which document and page your answer comes from.
3. Be concise. Use bullet points for multi-part answers.
4. Never make up facts not in the context."""


def build_context(results):
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, r in enumerate(results, 1):
        chunk = r["chunk"]
        parts.append(
            f"[{i}] File: {chunk['filename']} | Page: {chunk['page']}\n{chunk['text'][:600]}"
        )
    return "\n\n---\n\n".join(parts)


def ask(query, results):
    """
    Send the query + retrieved context to LLaMA-3 and get an answer.
    Returns a dict with answer, sources, latency, and token count.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    if not results:
        return {
            "answer": "No documents indexed yet. Upload a PDF first.",
            "sources": [],
            "latency_ms": 0,
            "tokens": 0,
            "has_answer": False,
        }

    context = build_context(results)
    user_msg = f"Context:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"

    t0 = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=800,
        temperature=TEMP,
    )
    ms = round((time.time() - t0) * 1000, 1)

    answer = response.choices[0].message.content.strip()
    no_answer_phrases = ["not in", "could not find", "don't have", "no information"]
    has_answer = not any(p in answer.lower() for p in no_answer_phrases)

    sources = [
        {
            "filename":   r["chunk"]["filename"],
            "page":       r["chunk"]["page"],
            "confidence": r["confidence"],
            "snippet":    r["chunk"]["text"][:180] + "...",
        }
        for r in results
    ]

    print(f"Answer generated in {ms}ms | tokens: {response.usage.total_tokens}")

    return {
        "answer":     answer,
        "sources":    sources,
        "latency_ms": ms,
        "tokens":     response.usage.total_tokens,
        "has_answer": has_answer,
    }