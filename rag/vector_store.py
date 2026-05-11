"""
vector_store.py — Lightweight in-memory vector store (replaces ChromaDB).

Stores embeddings as a numpy matrix and does brute-force cosine search.
Memory: ~8 MB for 5 000 chunks × 384 dims (float32). No external deps.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class VectorStore:
    _docs:  list[str]         = field(default_factory=list)
    _embs:  list[list[float]] = field(default_factory=list)
    _metas: list[dict]        = field(default_factory=list)

    def add(
        self,
        documents:  list[str],
        embeddings: list[list[float]],
        ids:        list[str],       # accepted for API compatibility, not stored
        metadatas:  list[dict],
    ) -> None:
        self._docs.extend(documents)
        self._embs.extend(embeddings)
        self._metas.extend(metadatas)

    def count(self) -> int:
        return len(self._docs)

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 5,
        include: list[str] | None = None,
    ) -> dict:
        if not self._embs:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        q   = np.array(query_embeddings[0], dtype=np.float32)
        mat = np.array(self._embs,          dtype=np.float32)

        # Cosine similarity — vectors are unit-normalised, so dot product == cosine
        scores = mat @ q
        n      = min(n_results, len(self._docs))
        idx    = np.argsort(scores)[::-1][:n]

        return {
            "documents": [[self._docs[i]  for i in idx]],
            "metadatas": [[self._metas[i] for i in idx]],
            "distances": [[float(1.0 - scores[i]) for i in idx]],
        }
