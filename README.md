# AskMyDocs — Production RAG Engine

Ask questions about any PDF. Get answers with exact page citations.

Built from scratch — no LangChain, no shortcuts.

## How it works
## What makes it different from basic RAG

| Basic RAG | AskMyDocs |
|---|---|
| Vector search only | BM25 + FAISS hybrid |
| No reranking | Cross-encoder reranker |
| One chunking strategy | Auto-selects by doc type |
| No evaluation | Confidence scores per result |

## Stack
Python · FastAPI · FAISS · BGE (BAAI/bge-base-en-v1.5) · BM25 · Cross-Encoder · Groq LLaMA-3

## Run it

```bash
git clone https://github.com/YOUR_USERNAME/Production-RAG-Engine
cd Production-RAG-Engine
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env
python3 -m uvicorn api.main:app --port 8000
```

Upload a PDF → ask questions → get answers with page numbers.

---
Built by Fardeen NS Khan | B.Tech CSE | NIT Surat
