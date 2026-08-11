import re

# tested 256 and 512, 400 worked best for my academic PDFs
# 80 overlap = 20% of chunk size — enough to not lose context at boundaries
CHUNK_SIZE = 400
OVERLAP    = 80


def fixed_chunks(text):
    """
    Split text into fixed-size chunks with overlap.
    Tries to end at a sentence boundary so we don't cut mid-sentence.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        # try to end at a period so we don't break a sentence
        if end < len(text):
            last_dot = text.rfind(".", start, end)
            if last_dot > start + CHUNK_SIZE // 2:
                end = last_dot + 1

        piece = text[start:end].strip()
        if len(piece) > 40:  # skip tiny leftover fragments
            chunks.append(piece)

        start = end - OVERLAP

    return chunks


def sentence_chunks(text):
    """
    Split by sentences. Good for legal text where every sentence matters.
    Groups 3 sentences per chunk with 1 sentence overlap.
    """
    # split on period/exclamation/question + space + capital letter
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = [s.strip() for s in re.split(pattern, text) if len(s.strip()) > 20]

    chunks = []
    window = 3   # sentences per chunk
    step   = 2   # move by 2 so adjacent chunks share 1 sentence

    for i in range(0, len(sentences), step):
        group = sentences[i:i + window]
        piece = " ".join(group).strip()
        if len(piece) > 40:
            chunks.append(piece)

    return chunks


def paragraph_chunks(text):
    """
    Split by paragraph breaks. Good for structured docs like manuals.
    Falls back to fixed chunking if paragraphs are too long.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
    chunks = []

    for para in paragraphs:
        if len(para) <= CHUNK_SIZE:
            chunks.append(para)
        else:
            # paragraph is too long, fall back to fixed size
            chunks.extend(fixed_chunks(para))

    return chunks


# which strategy to use for which doc type
STRATEGY = {
    "academic":  fixed_chunks,
    "legal":     sentence_chunks,
    "technical": paragraph_chunks,
    "general":   fixed_chunks,
}


def chunk_doc(doc):
    """
    Takes a doc dict (from loader.py) and returns a list of chunk dicts.
    Each chunk knows which file and page it came from — needed for citations.
    """
    strategy = STRATEGY[doc["doc_type"]]
    all_chunks = []
    idx = 0

    for page_data in doc["pages"]:
        page_num  = page_data["page"]
        page_text = page_data["text"]
        pieces    = strategy(page_text)

        for text in pieces:
            all_chunks.append({
                "id":       f"{doc['filename']}_{idx:04d}",
                "text":     text,
                "filename": doc["filename"],
                "page":     page_num,
                "doc_type": doc["doc_type"],
            })
            idx += 1

    print(f"  {doc['filename']} → {len(all_chunks)} chunks ({doc['doc_type']} strategy)")
    return all_chunks


def chunk_all(docs):
    """Chunk a list of docs, return all chunks in one flat list."""
    result = []
    for doc in docs:
        result.extend(chunk_doc(doc))
    print(f"Total chunks: {len(result)}")
    return result