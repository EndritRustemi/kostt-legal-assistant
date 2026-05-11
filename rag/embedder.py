"""
Shared embedding singleton — uses fastembed (ONNX Runtime, no PyTorch).

fastembed uses ~150 MB vs sentence-transformers/PyTorch which uses ~3-4 GB.
Both indexer.py and retriever.py import from here to share one model instance.

E5 models require explicit prefixes:
  "passage: ..." for document chunks
  "query: ..."   for search queries
"""
from __future__ import annotations
import numpy as np
from fastembed import TextEmbedding

EMBED_MODEL = "intfloat/multilingual-e5-small"   # 384-dim, multilingual, ONNX

_model: TextEmbedding | None = None


def get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(EMBED_MODEL)
    return _model


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def encode_passages(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed document chunks (adds 'passage: ' prefix required by E5)."""
    prefixed = ["passage: " + t for t in texts]
    model = get_model()
    return [
        _norm(np.array(e, dtype=np.float32)).tolist()
        for e in model.embed(prefixed, batch_size=batch_size)
    ]


def encode_query(query: str) -> list[float]:
    """Embed a search query (adds 'query: ' prefix required by E5)."""
    model = get_model()
    result = list(model.embed(["query: " + query], batch_size=1))
    return _norm(np.array(result[0], dtype=np.float32)).tolist()
