"""FAQ retrieval."""

from __future__ import annotations

from src.rag.store import load_index


def retrieve(query: str, k: int = 3):
    """Return top-k FAQ chunks for a customer query."""
    db = load_index()
    docs = db.similarity_search(query, k=k)
    return [
        {
            "content": d.page_content,
            "metadata": d.metadata,
            "score": getattr(d, "score", None),
        }
        for d in docs
    ]


def top_faq(query: str) -> dict | None:
    """Return the single best FAQ match."""
    results = retrieve(query, k=1)
    return results[0] if results else None
