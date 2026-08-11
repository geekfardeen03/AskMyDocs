import math
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import torch


RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def build_bm25(chunks):
    """
    BM25 is TF-IDF but smarter — handles word frequency saturation.
    Build once, search many times.
    """
    tokenized = [c["text"].lower().split() for c in chunks]
    return BM25Okapi(tokenized)


def bm25_search(bm25, chunks, query, top_k=20):
    """Keyword search. Returns (chunk, score) list."""
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top    = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(chunks[i], scores[i]) for i in top]


def reciprocal_rank_fusion(faiss_results, bm25_results, k=60):
    """
    RRF merges two ranked lists into one.
    Formula: score = sum of 1/(k + rank) across all lists.
    k=60 is from the original paper, smooths out rank differences.
    Returns sorted list of (chunk, rrf_score, faiss_rank, bm25_rank).
    """
    scores     = {}
    faiss_rank = {}
    bm25_rank  = {}
    chunk_map  = {}

    for rank, (chunk, _) in enumerate(faiss_results, 1):
        cid = chunk["id"]
        scores[cid]     = scores.get(cid, 0) + 1 / (k + rank)
        faiss_rank[cid] = rank
        chunk_map[cid]  = chunk

    for rank, (chunk, _) in enumerate(bm25_results, 1):
        cid = chunk["id"]
        scores[cid]    = scores.get(cid, 0) + 1 / (k + rank)
        bm25_rank[cid] = rank
        chunk_map[cid] = chunk

    ranked = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [
        (chunk_map[cid], scores[cid], faiss_rank.get(cid, 999), bm25_rank.get(cid, 999))
        for cid in ranked
    ]


def rerank(query, candidates, top_k=5):
    """
    Cross-encoder reads query+chunk TOGETHER — more accurate than vector similarity.
    Slower than FAISS but only runs on top 20, not all chunks.
    This is what gets you from "decent retrieval" to "good retrieval".
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = CrossEncoder(RERANKER, device=device)

    pairs  = [(query, c[0]["text"]) for c in candidates]
    raw_scores = model.predict(pairs, show_progress_bar=False)

    def sigmoid(x):
        return 1 / (1 + math.exp(-float(x)))

    # confidence = 70% reranker + 30% RRF position
    max_rrf = max(c[1] for c in candidates) or 1
    results = []
    for (chunk, rrf, fr, br), raw in zip(candidates, raw_scores):
        conf = 0.7 * sigmoid(raw) + 0.3 * (rrf / max_rrf)
        results.append({
            "chunk":       chunk,
            "confidence":  round(conf, 3),
            "faiss_rank":  fr,
            "bm25_rank":   br,
        })

    results.sort(key=lambda r: r["confidence"], reverse=True)
    return results[:top_k]


class Retriever:
    """
    Wraps the full pipeline: BM25 + FAISS → RRF → rerank.
    I keep BM25 and FAISS separate so I can update them independently
    when new documents come in.
    """

    def __init__(self, embedder):
        self.embedder = embedder
        self.bm25     = build_bm25(embedder.chunks) if embedder.chunks else None
        print(f"Retriever ready. BM25 over {len(embedder.chunks)} chunks.")

    def retrieve(self, query, top_k=5):
        if not self.embedder.chunks:
            return []

        faiss_res = self.embedder.search(query, top_k=20)
        bm25_res  = bm25_search(self.bm25, self.embedder.chunks, query, top_k=20)
        fused     = reciprocal_rank_fusion(faiss_res, bm25_res)
        final     = rerank(query, fused[:25], top_k=top_k)
        return final

    def update(self, new_chunks):
        """Call after adding new docs so BM25 sees them too."""
        all_chunks = self.embedder.chunks
        self.bm25  = build_bm25(all_chunks)
        print(f"BM25 rebuilt: {len(all_chunks)} chunks")