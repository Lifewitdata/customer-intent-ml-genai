"""Tests for RAG index and retrieval."""

from src.rag.faq import FAQ_DB, faq_documents
from src.rag.retriever import retrieve
from src.rag.store import build_index, save_index


def test_faq_db_non_empty():
    assert len(FAQ_DB) > 0
    docs = faq_documents()
    assert len(docs) == len(FAQ_DB)
    assert "page_content" in docs[0]
    assert "metadata" in docs[0]


def test_build_and_retrieve(tmp_path):
    db = build_index()
    save_index(db, tmp_path / "test_faiss")
    results = retrieve("my card is expiring", k=3)
    assert len(results) == 3
    assert any("card_about_to_expire" in r["content"] for r in results)
