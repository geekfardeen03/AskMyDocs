## Notebook 1 - Embedding Pipeline

Purpose:
Convert raw documents into vector embeddings.

Pipeline:

Document
    ↓
Paragraph Chunking
    ↓
Sliding Window Chunking
    ↓
SentenceTransformer
    ↓
384-D Embedding
    ↓
Save as Parquet