"""
indexer.py -- ChromaDB index builder for the Streamlit app.

Reads PDFs from the laws directory + optional web chunks,
embeds them with sentence-transformers, and returns a
ChromaDB collection ready for retrieval.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ─────────────────────────────────────────────────────────────────────

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

CATEGORIES: dict[str, str] = {
    # Primary folders — synced from HF Dataset
    "kosove":      "Legjislacioni i Kosovës",
    "zrre":        "Rregulloret ZRRE",
    "entso-e":     "Rregulloret ENTSO-E",
    "eu":          "Direktivat EU",
    "strategjike": "Dokumentet Strategjike",
    "vendime":     "Vendime",
    "te-tjera":    "Të Tjera",
    # Additional document folders
    "ligjet":      "Ligje (Civile & Procedurale)",
    "kontratat":   "Kontrata",
    "raportet":    "Raporte & Manuale",
}

_CHUNK_SIZE  = 600   # target characters per chunk
_CHUNK_OVER  = 80    # overlap between adjacent chunks

_model_cache: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model_cache
    if _model_cache is None:
        _model_cache = SentenceTransformer(EMBED_MODEL)
    return _model_cache


# ── PDF extraction ─────────────────────────────────────────────────────────────

def _extract_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for a PDF."""
    try:
        import fitz
        doc = fitz.open(str(path))
        pages = []
        for i, page in enumerate(doc, 1):
            text = page.get_text("text").strip()
            if text:
                pages.append((i, text))
        doc.close()
        return pages
    except Exception:
        return []


def _split_text(text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVER) -> list[str]:
    """Split text into overlapping chunks of approximately `size` characters."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        # Snap to word boundary
        if end < len(text):
            snap = text.rfind(" ", start, end)
            if snap > start:
                end = snap
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


def _chunk_id(source: str, page: int | str, idx: int) -> str:
    raw = f"{source}|{page}|{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


# ── Public API ─────────────────────────────────────────────────────────────────

def count_pdfs(laws_dir: Path) -> dict[str, int]:
    """Return {category_key: pdf_count} for all category sub-folders."""
    result: dict[str, int] = {}
    for key in CATEGORIES:
        folder = laws_dir / key
        result[key] = len(list(folder.glob("*.pdf"))) if folder.exists() else 0
    return result


def build_index(
    laws_dir: Path,
    extra_chunks: list[dict[str, Any]] | None = None,
) -> chromadb.Collection:
    """
    Build an in-memory ChromaDB collection from PDFs + optional web chunks.

    Parameters
    ----------
    laws_dir     : root folder containing per-category sub-folders with PDFs
    extra_chunks : web chunks returned by web_scraper.scrape_url(); each dict
                   must have keys: text, source, category, page, snippet

    Returns
    -------
    chromadb.Collection  ready for retriever.retrieve()
    """
    model  = _get_model()
    client = chromadb.EphemeralClient()
    col    = client.get_or_create_collection(
        "legal_docs",
        metadata={"hnsw:space": "cosine"},
    )

    texts, ids, metas = [], [], []

    # ── PDFs ──────────────────────────────────────────────────────────────────
    for cat_key, cat_label in CATEGORIES.items():
        folder = laws_dir / cat_key
        if not folder.exists():
            continue
        for pdf_path in sorted(folder.glob("*.pdf")):
            source = pdf_path.stem
            for page_num, page_text in _extract_pdf_pages(pdf_path):
                for idx, chunk_text in enumerate(_split_text(page_text)):
                    texts.append(chunk_text)
                    ids.append(_chunk_id(source, page_num, idx))
                    metas.append({
                        "source":   source,
                        "category": cat_label,
                        "page":     page_num,
                        "snippet":  chunk_text[:120],
                    })

    # ── Web chunks ────────────────────────────────────────────────────────────
    for chunk in (extra_chunks or []):
        text = chunk.get("text", "").strip()
        if not text:
            continue
        src  = str(chunk.get("source", "web"))
        page = chunk.get("page", "web")
        idx  = chunk.get("_idx", 0)
        texts.append(text)
        ids.append(_chunk_id(src, page, idx))
        metas.append({
            "source":   src,
            "category": str(chunk.get("category", "Web")),
            "page":     str(page),
            "snippet":  chunk.get("snippet", text[:120]),
        })

    if not texts:
        return col

    # ── Embed + add in batches of 100 ────────────────────────────────────────
    embeddings = model.encode(texts, convert_to_numpy=True,
                              normalize_embeddings=True,
                              show_progress_bar=False).tolist()
    batch = 100
    for i in range(0, len(texts), batch):
        col.add(
            documents=embeddings[i : i + batch],
            embeddings=embeddings[i : i + batch],
            ids=ids[i : i + batch],
            metadatas=metas[i : i + batch],
        )

    return col
