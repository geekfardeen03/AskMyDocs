import sys
import os
import shutil
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# add src to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from loader    import load_pdf
from chunker   import chunk_doc
from embedder  import Embedder
from retriever import Retriever
from generator import ask

# global objects — loaded once at startup, reused for every request
emb = None
ret = None

# simple stats I track to show in the health endpoint
query_count  = 0
total_lat_ms = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global emb, ret
    print("Starting AskMyDocs API...")
    emb = Embedder()
    loaded = emb.load()
    if loaded and emb.chunks:
        ret = Retriever(emb)
        print(f"Ready — {emb.total_docs} docs indexed")
    else:
        print("No index yet. Upload a PDF via /upload")
    yield
    print("Shutting down.")


app = FastAPI(title="AskMyDocs", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "https://ask-my-docs-six.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)    


# ── Request / Response models ─────────────────────────────────────────────────

class QueryReq(BaseModel):
    query: str
    top_k: int = 5

class QueryRes(BaseModel):
    query:      str
    answer:     str
    sources:    list
    has_answer: bool
    latency_ms: float
    tokens:     int


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "docs_indexed": emb.total_docs if emb else 0,
        "chunks_total": len(emb.chunks) if emb and emb.chunks else 0,
        "queries_served": query_count,
        "avg_latency_ms": round(total_lat_ms / query_count, 1) if query_count else 0,
    }


@app.post("/query", response_model=QueryRes)
def query(req: QueryReq):
    global query_count, total_lat_ms, ret

    if not emb or not emb.chunks:
        raise HTTPException(503, "No documents indexed. Upload a PDF first.")
    if ret is None:
        raise HTTPException(503, "Retriever not ready.")

    t0      = time.time()
    results = ret.retrieve(req.query, top_k=req.top_k)
    ans     = ask(req.query, results)
    ms      = round((time.time() - t0) * 1000, 1)

    query_count  += 1
    total_lat_ms += ms

    return QueryRes(
        query      = req.query,
        answer     = ans["answer"],
        sources    = ans["sources"],
        has_answer = ans["has_answer"],
        latency_ms = ms,
        tokens     = ans["tokens"],
    )


def _index_pdf(path: str):
    """Runs in background after upload."""
    global ret
    doc = load_pdf(path)
    if not doc:
        return
    chunks = chunk_doc(doc)
    emb.add_to_index(chunks)
    if ret is None:
        ret = Retriever(emb)
    else:
        ret.update(chunks)
    print(f"Indexed: {doc['filename']} ({len(chunks)} chunks)")


@app.post("/upload")
async def upload(bg: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDFs supported.")

    save_path = os.path.join("data", "documents", file.filename)
    os.makedirs("data/documents", exist_ok=True)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    bg.add_task(_index_pdf, save_path)

    return {"message": f"'{file.filename}' uploaded. Indexing in background..."}


@app.get("/documents")
def documents():
    if not emb or not emb.chunks:
        return {"documents": []}

    seen = {}
    for c in emb.chunks:
        fn = c["filename"]
        if fn not in seen:
            seen[fn] = {"filename": fn, "doc_type": c["doc_type"], "chunks": 0}
        seen[fn]["chunks"] += 1

    return {"documents": list(seen.values())}