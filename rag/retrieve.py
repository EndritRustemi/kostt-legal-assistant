"""
retrieve.py — Hybrid Legal Chunk Retriever

Retrieves relevant legal chunks using a combination of:
  - Semantic search  (sentence-transformers cosine similarity)
  - Keyword search   (BM25 — Okapi BM25)
  - Query expansion  (query_expander.py — Albanian/English legal terminology)

Retrieval flow
--------------
1. Expand the user query into 3-5 variants (query_expander.py).
2. Run hybrid retrieval independently for each variant.
3. Merge all results, keep the highest score per chunk id.
4. Re-rank by score and return the top-k chunks.

Query expansion requires an Anthropic API key passed to retrieve().
If no key is supplied, a single-query retrieval is performed (no expansion).

Only chunks carrying the required legal metadata fields are indexed.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np

from rag.debug import dprint

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL     = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_TOP_K     = 5
DEFAULT_THRESHOLD = 0.5   # minimum hybrid score to include a chunk
DEFAULT_ALPHA     = 0.7   # weight for semantic; (1-alpha) goes to keyword

# A legal chunk must have at least these fields
LEGAL_FIELDS = frozenset({"id", "text", "source", "article"})

# Module-level model cache (avoid reloading across calls)
_MODEL_CACHE: dict[str, Any] = {}


# ── BM25 ───────────────────────────────────────────────────────────────────────

class _BM25:
    """
    Okapi BM25 with standard defaults k1=1.5, b=0.75.
    Operates on pre-tokenised documents.
    """

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1     = k1
        self.b      = b
        self.n      = len(corpus)
        self.avgdl  = sum(len(d) for d in corpus) / max(self.n, 1)
        self.tf     = [Counter(doc) for doc in corpus]
        self.df: dict[str, int] = {}
        for doc in corpus:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        """Return a BM25 score vector (length = corpus size) for the query."""
        out = np.zeros(self.n, dtype=float)
        for term in set(query_tokens):
            if term not in self.df:
                continue
            idf = math.log(
                (self.n - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1.0
            )
            for i, tf_doc in enumerate(self.tf):
                freq = tf_doc.get(term, 0)
                if freq == 0:
                    continue
                dl = sum(tf_doc.values())
                tf_norm = freq * (self.k1 + 1) / (
                    freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
                out[i] += idf * tf_norm
        return out


# ── Utilities ──────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric (Latin + extended Latin)."""
    return re.findall(r"[\wÀ-ɏ]+", text.lower())


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _minmax(arr: np.ndarray) -> np.ndarray:
    """Scale array to [0, 1]. Returns all-zeros if array is constant."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _load_model(name: str) -> Any:
    if name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        _MODEL_CACHE[name] = SentenceTransformer(name)
    return _MODEL_CACHE[name]


def _is_legal(chunk: dict) -> bool:
    """Return True only if chunk has all required legal metadata and non-empty text."""
    return LEGAL_FIELDS.issubset(chunk.keys()) and bool(str(chunk.get("text", "")).strip())


# ── Core class ─────────────────────────────────────────────────────────────────

class LegalRetriever:
    """
    Hybrid (semantic + keyword) retriever for legal document chunks.

    Usage
    -----
    retriever = LegalRetriever(chunks)          # build index once
    results   = retriever.retrieve("query")     # call as many times as needed
    """

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        embed_model: str = DEFAULT_MODEL,
        alpha: float = DEFAULT_ALPHA,
    ) -> None:
        """
        Parameters
        ----------
        chunks : list[dict]
            Corpus to index. Each dict must contain id, text, source, article.
            Chunks missing any of these are silently skipped.
        embed_model : str
            Sentence-transformers model name.
        alpha : float
            Hybrid blend weight in [0, 1].
            1.0 = pure semantic; 0.0 = pure keyword.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.alpha       = alpha
        self._model_name = embed_model

        # Filter to legal-only chunks
        self._chunks = [c for c in chunks if _is_legal(c)]
        self._ready  = bool(self._chunks)

        if not self._ready:
            return

        texts = [str(c["text"]) for c in self._chunks]

        # Semantic index
        model = _load_model(embed_model)
        self._embeddings: np.ndarray = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Keyword index
        self._bm25 = _BM25([_tokenize(t) for t in texts])

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = DEFAULT_THRESHOLD,
        api_key: str = "",
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top legal chunks for *query* using expanded queries.

        Parameters
        ----------
        query : str
            The search query (natural language or legal keywords).
        top_k : int
            Maximum number of results. Default 5.
        threshold : float
            Minimum hybrid score in [0, 1]. Chunks below this are excluded.
        api_key : str
            Anthropic API key for query expansion. If empty, a single-query
            retrieval is performed without expansion.

        Returns
        -------
        list[dict]
            Chunks enriched with a "score" key (float, 4 dp), ranked highest
            first. Returns [] if nothing passes the threshold.
        """
        if not query or not query.strip():
            return []
        if not self._ready:
            return []

        # ── Step 1: Expand the query ───────────────────────────────────────────
        queries: list[str] = [query]
        if api_key:
            try:
                from rag.query_expander import expand as _expand
                expansion = _expand(query, api_key)
                queries = expansion.queries
            except Exception as exc:
                dprint(f"Query expansion failed ({exc}), using original query")

        dprint("Expanded queries", queries)

        # ── Step 2: Retrieve for each expanded query, then merge ──────────────
        seen: dict[str, dict[str, Any]] = {}   # chunk_id -> best chunk

        for q in queries:
            for chunk in self._retrieve_single(q, threshold):
                cid = str(chunk["id"])
                if cid not in seen or chunk["score"] > seen[cid]["score"]:
                    seen[cid] = chunk

        # ── Step 3: Re-rank by score and clip to top_k ────────────────────────
        results = sorted(seen.values(), key=lambda c: c["score"], reverse=True)[:top_k]

        dprint(
            "Chunk scores",
            [
                f"{c.get('id', 'unknown'):<20} score={c['score']:.4f}  "
                f"{c.get('source', '')} -- {c.get('article', '')}"
                for c in results
            ],
        )

        return results

    # ── Internal single-query retrieval ────────────────────────────────────────

    def _retrieve_single(
        self,
        query: str,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Run one hybrid retrieval pass and return all chunks above threshold."""
        # 1. Semantic scores
        model  = _load_model(self._model_name)
        q_emb  = model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sem_raw  = np.array([_cosine(q_emb, emb) for emb in self._embeddings])
        sem_norm = _minmax(sem_raw)

        # 2. Keyword (BM25) scores
        q_tokens = _tokenize(query)
        kw_raw   = self._bm25.scores(q_tokens)
        kw_norm  = _minmax(kw_raw)

        # 3. Hybrid fusion
        hybrid = self.alpha * sem_norm + (1.0 - self.alpha) * kw_norm

        # 4. Filter by threshold (no top_k clip — caller handles that)
        results: list[dict[str, Any]] = []
        for i in range(len(self._chunks)):
            score = float(hybrid[i])
            if score >= threshold:
                chunk = dict(self._chunks[i])   # shallow copy
                chunk["score"] = round(score, 4)
                results.append(chunk)

        return results

    @property
    def chunk_count(self) -> int:
        """Number of valid legal chunks currently indexed."""
        return len(self._chunks)

    def __repr__(self) -> str:
        return (
            f"LegalRetriever(chunks={self.chunk_count}, "
            f"model='{self._model_name}', alpha={self.alpha}, "
            f"ready={self._ready})"
        )


# ── Stateless convenience wrapper ──────────────────────────────────────────────

def retrieve(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
    alpha: float = DEFAULT_ALPHA,
    embed_model: str = DEFAULT_MODEL,
    api_key: str = "",
) -> list[dict[str, Any]]:
    """
    One-shot hybrid retrieval with optional query expansion.

    Builds a LegalRetriever on-the-fly and returns results.
    For repeated queries on the same corpus, instantiate LegalRetriever
    directly to avoid rebuilding the index each time.

    Parameters
    ----------
    api_key : str
        Gemini API key for query expansion. Pass "" to skip expansion.

    Returns
    -------
    list[dict]
        [
          {
            "id":        "...",
            "text":      "...",
            "source":    "...",
            "article":   "...",
            "paragraph": "...",   # if present in original chunk
            "score":     0.82
          },
          ...
        ]
        Empty list if no chunks pass the threshold.
    """
    return LegalRetriever(chunks, embed_model=embed_model, alpha=alpha).retrieve(
        query, top_k=top_k, threshold=threshold, api_key=api_key
    )


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys

    _corpus = [
        {
            "id": "k1",
            "source": "Law No. 05/L-085 on Electricity",
            "article": "Article 5",
            "paragraph": "Paragraph 1",
            "text": (
                "KOSTT, as the transmission system operator, is responsible for the "
                "secure, reliable and efficient operation of the transmission network "
                "in Kosovo, ensuring non-discriminatory access to all users."
            ),
        },
        {
            "id": "k2",
            "source": "Law No. 05/L-085 on Electricity",
            "article": "Article 12",
            "paragraph": "Paragraph 3",
            "text": (
                "The transmission system operator shall publish the connection terms "
                "and conditions, including technical and financial criteria, and shall "
                "not discriminate between users seeking access to the network."
            ),
        },
        {
            "id": "k3",
            "source": "ZRRE Network Code",
            "article": "Article 8",
            "paragraph": "Paragraph 2",
            "text": (
                "Connection requests must be processed within 30 days of receipt. "
                "Rejection must be justified in writing and may be appealed to the "
                "Energy Regulatory Office."
            ),
        },
        {
            "id": "k4",
            "source": "ZRRE Market Rules",
            "article": "Article 3",
            "paragraph": "Paragraph 1",
            "text": (
                "All market participants must register with KOSTT before engaging in "
                "electricity trading activities. Registration requires submission of "
                "technical and financial capability documentation."
            ),
        },
        {   # intentionally missing 'article' — must be filtered out
            "id": "bad1",
            "source": "Some Doc",
            "text": "This chunk has no article field and should be excluded.",
        },
    ]

    _api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    print("Building LegalRetriever index...")
    retriever = LegalRetriever(_corpus)
    print(retriever, "\n")

    # ── Test A: with query expansion (requires API key) ────────────────────────
    _query = "A mund te refuzoje KOSTT nje kerkese per lidhje ne rrjet?"
    print("=" * 60)
    print(f"TEST A — query expansion | api_key={'set' if _api_key else 'NOT SET'}")
    print(f"Query: {_query!r}")
    print("=" * 60)
    _results = retriever.retrieve(_query, top_k=5, threshold=DEFAULT_THRESHOLD,
                                  api_key=_api_key)
    if _results:
        for r in _results:
            print(f"  [score={r['score']}] {r['source']} -- {r['article']}")
    else:
        print("  No chunks above threshold.")

    # ── Test B: no expansion (no api_key) ─────────────────────────────────────
    print()
    print("=" * 60)
    print("TEST B — no expansion (api_key='')")
    print("=" * 60)
    _results2 = retriever.retrieve(
        "Can KOSTT reject a connection request?",
        top_k=3, threshold=DEFAULT_THRESHOLD, api_key="",
    )
    if _results2:
        for r in _results2:
            print(f"  [score={r['score']}] {r['source']} -- {r['article']}")
    else:
        print("  No chunks above threshold.")

    # Verify bad chunk was excluded in both runs
    for run in (_results, _results2):
        assert all(r["id"] != "bad1" for r in run), "Non-legal chunk leaked through!"

    print()
    print("Non-legal chunks correctly filtered out.")
    print("Smoke test complete.")
