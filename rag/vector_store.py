"""
vector_store.py — BM25-based document store (replaces chromadb + fastembed).

BM25 (Best Match 25) is a keyword retrieval algorithm that requires:
  - No ML model, no ONNX Runtime, no PyTorch
  - ~5 MB RAM for the index (vs 300-500 MB for an embedding model)
  - No network download at startup

For legal documents with precise terminology (article numbers, law names,
specific legal terms) BM25 retrieval quality is comparable to semantic search.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """Lowercase + split on non-word chars. Handles ë, ç and other Unicode."""
    return re.findall(r"\w+", text.lower())


@dataclass
class VectorStore:
    """BM25 document store with the same .add()/.query()/.count() interface
    as the previous numpy VectorStore — app.py needs no changes."""

    _docs:  list[str]  = field(default_factory=list)
    _metas: list[dict] = field(default_factory=list)
    _bm25:  Any        = field(default=None, repr=False)

    # ── write ──────────────────────────────────────────────────────────────────

    def add(
        self,
        documents:  list[str],
        embeddings: list[Any],   # ignored — kept for API compatibility
        ids:        list[str],   # ignored
        metadatas:  list[dict],
    ) -> None:
        self._docs.extend(documents)
        self._metas.extend(metadatas)
        self._bm25 = None        # invalidate; rebuilt lazily on next query

    def count(self) -> int:
        return len(self._docs)

    # ── read ───────────────────────────────────────────────────────────────────

    def _ensure_index(self) -> None:
        if self._bm25 is None and self._docs:
            corpus = [_tokenize(d) for d in self._docs]
            self._bm25 = BM25Okapi(corpus)

    def query(
        self,
        query_embeddings: list[Any] | None = None,  # ignored
        query: str = "",
        n_results: int = 5,
        include: list[str] | None = None,
    ) -> dict:
        if not self._docs:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        self._ensure_index()
        tokens = _tokenize(query)
        if not tokens:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        scores = self._bm25.get_scores(tokens)           # numpy array
        max_s  = float(scores.max()) if scores.max() > 0 else 1.0
        norm   = scores / max_s                          # normalise to [0,1]

        n   = min(n_results, len(self._docs))
        idx = np.argsort(scores)[::-1][:n]

        return {
            "documents": [[self._docs[i]  for i in idx]],
            "metadatas": [[self._metas[i] for i in idx]],
            # distance = 1 - similarity; retriever converts back with 1-dist
            "distances": [[float(1.0 - norm[i]) for i in idx]],
        }
