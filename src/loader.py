import os
import pymupdf as fitz


# how i detect what kind of document it is
# just keyword matching - simple and fast, no model needed
ACADEMIC_WORDS = ["abstract", "introduction", "references", "doi", "methodology"]
LEGAL_WORDS    = ["whereas", "plaintiff", "court", "judgment", "clause", "section"]
TECH_WORDS     = ["installation", "api", "endpoint", "configuration", "parameter"]


def detect_type(text_sample):
    t = text_sample.lower()
    scores = {
        "academic": sum(1 for w in ACADEMIC_WORDS if w in t),
        "legal":    sum(1 for w in LEGAL_WORDS if w in t),
        "technical": sum(1 for w in TECH_WORDS if w in t),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "general"


def load_pdf(path):
    """
    Load one PDF. Returns a dict with filename, pages, and doc type.
    I keep pages separate because I need page numbers for citations.
    """
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None

    try:
        pdf = fitz.open(path)
    except Exception as e:
        print(f"Couldn't open {path}: {e}")
        return None

    pages = []
    for i in range(len(pdf)):
        text = pdf[i].get_text("text").strip()
        if text:  # skip blank pages
            pages.append({"page": i + 1, "text": text})

    pdf.close()

    if not pages:
        print(f"No text found in {path} - might be scanned")
        return None

    # check first page to figure out doc type
    first_page = pages[0]["text"]
    doc_type = detect_type(first_page)

    print(f"Loaded: {os.path.basename(path)} | {len(pages)} pages | type={doc_type}")

    return {
        "filename": os.path.basename(path),
        "path":     path,
        "doc_type": doc_type,
        "pages":    pages,
    }


def load_folder(folder):
    """Load all PDFs from a folder."""
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        return []

    pdfs = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    print(f"Found {len(pdfs)} PDF(s) in {folder}")

    docs = []
    for name in pdfs:
        doc = load_pdf(os.path.join(folder, name))
        if doc:
            docs.append(doc)

    print(f"Loaded {len(docs)}/{len(pdfs)} successfully")
    return docs