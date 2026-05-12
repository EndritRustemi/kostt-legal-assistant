"""
vector_store.py — Memory-efficient BM25 with an inverted index.

rank_bm25 tokenises ALL documents into one giant list before building
the index.  For 400 k+ chunks that list alone can reach 5-8 GB and
triggers the HF Spaces 16 Gi OOM.

This implementation processes one document at a time, so peak RAM is
≈ (original text) + (one tokenised doc) + (inverted index) — typically
< 400 MB for the full legal corpus.

Algorithm
---------
  - Inverted index: term → [(doc_id, tf), ...]
  - IDF:  log(1 + (N - df + 0.5) / (df + 0.5))        (same as BM25Okapi)
  - Score: idf * tf*(k1+1) / (tf + k1*(1-b + b*dl/avgdl))
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

K1 = 1.5
B  = 0.75


def _tokenize(text: str) -> list[str]:
    """Lowercase + split on non-word chars.  Handles ë, ç and Unicode."""
    return re.findall(r"\w+", text.lower())


@dataclass
class VectorStore:
    """Inverted-index BM25 store.

    Same .add() / .query() / .count() interface as before — app.py
    and retriever.py need zero changes.
    """

    _docs:   list[str]  = field(default_factory=list)
    _metas:  list[dict] = field(default_factory=list)

    # Inverted index built incrementally (one doc at a time)
    _index:          dict = field(default_factory=dict)   # term -> [(doc_id, tf)]
    _doc_len:        list = field(default_factory=list)   # token count per doc
    _indexed_up_to:  int  = field(default=0)
    _idf:            dict = field(default_factory=dict)   # term -> float
    _avgdl:          float = field(default=0.0)
    _idf_dirty:      bool  = field(default=True)          # recompute IDF after adds

    # ── write ──────────────────────────────────────────────────────────────────

    def add(
        self,
        documents:  list[str],
        embeddings: list[Any],   # ignored
        ids:        list[str],   # ignored
        metadatas:  list[dict],
    ) -> None:
        self._docs.extend(documents)
        self._metas.extend(metadatas)
        self._idf_dirty = True   # IDF needs recalculation after new docs

    def count(self) -> int:
        return len(self._docs)

    # ── internal index build ──────────────────────────────────────────────────

    def _build_index(self) -> None:
        """Incrementally index any new documents since last call.

        Processes one document at a time — peak extra RAM is O(one doc's
        tokens), not O(all tokens), so there is no bulk list allocation.
        """
        n = len(self._docs)
        for i in range(self._indexed_up_to, n):
            tokens = _tokenize(self._docs[i])
            self._doc_len.append(len(tokens))

            tf = Counter(tokens)           # local to this iteration → freed immediately
            for term, count in tf.items():
                if term not in self._index:
                    self._index[term] = []
                self._index[term].append((i, count))

        self._indexed_up_to = n

        if n > 0:
            self._avgdl = sum(self._doc_len) / n

        # Recompute IDF
        for term, postings in self._index.items():
            df = len(postings)
            self._idf[term] = math.log(1 + (n - df + 0.5) / (df + 0.5))

        self._idf_dirty = False

    # ── read ───────────────────────────────────────────────────────────────────

    def query(
        self,
        query_embeddings: list[Any] | None = None,  # ignored
        query: str = "",
        n_results: int = 5,
        include: list[str] | None = None,
    ) -> dict:
        if not self._docs:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        if self._idf_dirty or self._indexed_up_to < len(self._docs):
            self._build_index()

        tokens = set(_tokenize(query))
        if not tokens:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        n      = len(self._docs)
        scores = np.zeros(n, dtype=np.float32)
        avgdl  = self._avgdl or 1.0

        for term in tokens:
            if term not in self._index:
                continue
            idf = self._idf.get(term, 0.0)
            for doc_id, tf in self._index[term]:
                dl   = self._doc_len[doc_id]
                norm = K1 * (1.0 - B + B * dl / avgdl)
                scores[doc_id] += idf * tf * (K1 + 1) / (tf + norm)

        k     = min(n_results, n)
        max_s = float(scores.max()) if scores.max() > 0 else 1.0
        normed = scores / max_s

        idx = np.argsort(scores)[::-1][:k]
        return {
            "documents": [[self._docs[i]  for i in idx]],
            "metadatas": [[self._metas[i] for i in idx]],
            # distance = 1 - similarity; retriever converts back with 1-dist
            "distances": [[float(1.0 - normed[i]) for i in idx]],
        }
