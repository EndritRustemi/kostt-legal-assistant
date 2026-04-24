"""
Kërkim vektorial: embedo pyetjen (lokal) → gjej chunks më të ngjashme
"""

import chromadb
from sentence_transformers import SentenceTransformer
from rag.ingest import EMBED_MODEL

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def retrieve(col: chromadb.Collection, question: str, api_key: str = "", top_k: int = 5) -> list[dict]:
    model = _get_model()
    query_embedding = model.encode([question])[0].tolist()

    results = col.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta.get("source", ""),
            "category": meta.get("category", ""),
            "page": meta.get("page", 0),
            "snippet": meta.get("snippet", doc[:120]),
            "score": round(1 - dist, 3),
        })

    return chunks
