"""Kërkim BM25: gjej chunks më relevante për pyetjen."""

from rag.vector_store import VectorStore


def retrieve(store: VectorStore, question: str, api_key: str = "", top_k: int = 5) -> list[dict]:
    if store.count() == 0:
        return []

    results = store.query(
        query=question,
        n_results=min(top_k, store.count()),
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
            "score":    round(1.0 - dist, 3),
        })

    return chunks
