
import os
import pickle
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

# BAAI/bge gives better retrieval quality than MiniLM in our tests.
# 768 dimensions vs 384 — worth the extra memory for better retrieval.
MODEL_NAME = "BAAI/bge-base-en-v1.5"
DIM = 768
BATCH = 64   # safe for 6GB VRAM

INDEX_FILE = "indexes/faiss.index"
CHUNKS_FILE = "indexes/chunks.pkl"


class Embedder:
    def __init__(self):
        # Lazy loading: don't load the model when FastAPI starts.
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.index = None
        self.chunks = []

        print(
            f"Embedder created. "
            f"Model will load on first use ({self.device.upper()})."
        )

    def _ensure_model(self):
        """Load the embedding model only when it is actually needed."""
        if self.model is None:
            print(f"Loading embedding model on {self.device.upper()}...")

            self.model = SentenceTransformer(
                MODEL_NAME,
                device=self.device
            )

            print("Model ready.")

    def embed(self, texts):
        """
        Convert a list of strings to vectors.

        normalize_embeddings=True means cosine similarity = dot product.
        BGE also needs a special prefix for queries (not for documents).
        """
        self._ensure_model()

        return self.model.encode(
            texts,
            batch_size=BATCH,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def embed_query(self, query):
        """
        Queries need a prefix with BGE — it was trained this way.
        Without it, recall drops about 5-8% in my tests.
        """
        self._ensure_model()

        prefixed = (
            "Represent this sentence for searching relevant passages: "
            f"{query}"
        )

        vec = self.model.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return vec.astype(np.float32)

    def build_index(self, chunks):
        """
        Build FAISS index from a list of chunk dicts.

        Using IndexFlatIP (exact search with inner product) — fine for
        <50k chunks. For larger sets I'd switch to IVF but this keeps
        it simple.
        """
        self._ensure_model()

        print(f"Embedding {len(chunks)} chunks...")

        texts = [c["text"] for c in chunks]
        vecs = self.embed(texts)

        self.index = faiss.IndexFlatIP(DIM)
        self.index.add(vecs)
        self.chunks = chunks

        self.save()

        print(
            f"Index built: {self.index.ntotal} vectors saved."
        )

    def add_to_index(self, new_chunks):
        """Add more chunks without rebuilding from scratch."""
        texts = [c["text"] for c in new_chunks]
        vecs = self.embed(texts)

        if self.index is None:
            self.index = faiss.IndexFlatIP(DIM)

        self.index.add(vecs)
        self.chunks.extend(new_chunks)
        self.save()

        print(
            f"Added {len(new_chunks)} chunks. "
            f"Total: {self.index.ntotal}"
        )

    def search(self, query, top_k=20):
        """Return top_k (chunk, score) tuples for a query."""
        if self.index is None or self.index.ntotal == 0:
            return []

        qvec = self.embed_query(query)
        scores, indices = self.index.search(qvec, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append(
                    (self.chunks[idx], float(score))
                )

        return results

    def save(self):
        os.makedirs("indexes", exist_ok=True)

        faiss.write_index(
            self.index,
            INDEX_FILE
        )

        with open(CHUNKS_FILE, "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self):
        """Load saved index. Returns True if successful."""
        if not os.path.exists(INDEX_FILE):
            return False

        self.index = faiss.read_index(INDEX_FILE)

        with open(CHUNKS_FILE, "rb") as f:
            self.chunks = pickle.load(f)

        print(
            f"Loaded index: {self.index.ntotal} vectors, "
            f"{len(set(c['filename'] for c in self.chunks))} docs"
        )

        return True

    @property
    def total_docs(self):
        return (
            len(set(c["filename"] for c in self.chunks))
            if self.chunks
            else 0
        )
EOF