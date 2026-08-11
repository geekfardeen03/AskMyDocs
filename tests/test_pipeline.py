"""
Quick sanity checks for each module.
Run with: python3 tests/test_pipeline.py
All 5 should pass before you start the API.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_chunker_fixed():
    from chunker import fixed_chunks
    text   = "Hello world this is a test sentence. " * 30
    chunks = fixed_chunks(text)
    assert len(chunks) > 1, "should produce multiple chunks"
    assert all(len(c) > 20 for c in chunks), "no tiny chunks"
    print("  PASS — fixed_chunks")


def test_chunker_sentence():
    from chunker import sentence_chunks
    text   = ". ".join([f"This is sentence number {i}" for i in range(30)]) + "."
    chunks = sentence_chunks(text)
    assert len(chunks) > 1, "should produce multiple chunks"
    print("  PASS — sentence_chunks")


def test_doc_type():
    from loader import detect_type
    assert detect_type("abstract introduction references doi methodology") == "academic"
    assert detect_type("whereas plaintiff court judgment clause") == "legal"
    assert detect_type("installation api endpoint configuration parameter") == "technical"
    print("  PASS — detect_type")


def test_embedder_basic():
    from embedder import Embedder

    e = Embedder()
    dummy = [
        {"id": f"test_{i}", "text": f"This is test chunk {i} about AI and ML.", 
         "filename": "test.pdf", "page": 1, "doc_type": "general"}
        for i in range(5)
    ]
    e.build_index(dummy)
    assert e.index is not None
    assert e.index.ntotal == 5

    results = e.search("machine learning", top_k=3)
    assert len(results) > 0
    assert results[0][1] > 0  # score should be positive
    print(f"  PASS — embedder (5 vectors, got {len(results)} results)")


def test_query_embed():
    from embedder import Embedder
    e   = Embedder()
    vec = e.embed_query("what is retrieval augmented generation?")
    assert vec.shape == (1, 768), f"wrong shape: {vec.shape}"
    print(f"  PASS — query embedding shape {vec.shape}")


if __name__ == "__main__":
    print("\nAskMyDocs — Test Suite")
    print("=" * 35)
    tests = [
        test_chunker_fixed,
        test_chunker_sentence,
        test_doc_type,
        test_query_embed,
        test_embedder_basic,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL — {t.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} passed")
    print("=" * 35)