"""Build and save the FAISS vector store over the FAQ knowledge base."""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.rag.faq import faq_documents

ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT / "faiss_index"


def build_index(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
    docs = faq_documents()
    texts = [d["page_content"] for d in docs]
    metadatas = [d["metadata"] for d in docs]
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return db


def save_index(db: FAISS, path: Path | None = None) -> None:
    path = path or INDEX_DIR
    path.parent.mkdir(parents=True, exist_ok=True)
    db.save_local(str(path))


def load_index(path: Path | None = None) -> FAISS:
    path = path or INDEX_DIR
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)


def main():
    print("Building FAQ vector index...")
    db = build_index()
    save_index(db)
    print(f"Saved FAISS index to {INDEX_DIR}")


if __name__ == "__main__":
    main()
