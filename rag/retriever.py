"""Kërkim vektorial: embedo pyetjen → gjej chunks më të ngjashme."""

from rag.embedder import encode_query
from rag.vector_store import VectorStore


def retrieve(store: VectorStore, question: str, api_key: str = "", top_k: int = 5) -> list[dict]:
    query_embedding = encode_query(question)

    results = store.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, store.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":     doc,
            "source":   meta.get("source", ""),
            "category": meta.get("category", ""),
            "page":     meta.get("page", 0),
            "snippet":  meta.get("snippet", doc[:120]),
            "score":    round(1 - dist, 3),
        })

    return chunks
