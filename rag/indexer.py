"""
indexer.py -- Builds the in-memory VectorStore for the Streamlit app.

Memory-safe: each PDF is embedded and added individually so peak memory
is bounded by a single document's chunks, not the whole corpus.

The VectorStore is cached by @st.cache_resource in app.py and survives
for the lifetime of the Streamlit process. Clearing the cache (e.g. via
"Ri-indekso" button) triggers a full rebuild on next call.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from rag.embedder import encode_passages, EMBED_MODEL
from rag.vector_store import VectorStore

# ── Config ─────────────────────────────────────────────────────────────────────

CATEGORIES: dict[str, str] = {
    "kosove":      "Legjislacioni i Kosovës",
    "zrre":        "Rregulloret ZRRE",
    "entso-e":     "Rregulloret ENTSO-E",
    "eu":          "Direktivat EU",
    "strategjike": "Dokumentet Strategjike",
    "vendime":     "Vendime",
    "te-tjera":    "Të Tjera",
    "kontratat":   "Kontrata",
    "raportet":    "Raporte & Manuale",
}

_CHUNK_SIZE = 800
_CHUNK_OVER = 100

# ── Helpers ─────────────────────────────────────────────────────────────────────

def _extract_pdf_pages(path: Path) -> list[tuple[int, str]]:
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
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            snap = text.rfind(" ", start, end)
            if snap > start:
                end = snap
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


def _chunk_id(source: str, page: int | str, idx: int) -> str:
    return hashlib.md5(f"{source}|{page}|{idx}".encode()).hexdigest()[:16]


# ── Public API ─────────────────────────────────────────────────────────────────

def count_pdfs(laws_dir: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for key in CATEGORIES:
        folder = laws_dir / key
        result[key] = len(list(folder.glob("*.pdf"))) if folder.exists() else 0
    return result


def build_index(
    laws_dir: Path,
    extra_chunks: list[dict[str, Any]] | None = None,
) -> VectorStore:
    """
    Build an in-memory VectorStore from all PDFs + optional web chunks.
    Each PDF is embedded individually to keep peak memory minimal.
    """
    store = VectorStore()

    # ── PDFs — one at a time ──────────────────────────────────────────────────
    for cat_key, cat_label in CATEGORIES.items():
        folder = laws_dir / cat_key
        if not folder.exists():
            continue
        for pdf_path in sorted(folder.glob("*.pdf")):
            source = pdf_path.stem
            texts:  list[str]  = []
            ids:    list[str]  = []
            metas:  list[dict] = []

            for page_num, page_text in _extract_pdf_pages(pdf_path):
                for idx, chunk in enumerate(_split_text(page_text)):
                    texts.append(chunk)
                    ids.append(_chunk_id(source, page_num, idx))
                    metas.append({
                        "source":   source,
                        "category": cat_label,
                        "page":     page_num,
                        "snippet":  chunk[:120],
                    })

            if not texts:
                continue

            embeddings = encode_passages(texts, batch_size=32)
            batch = 100
            for i in range(0, len(texts), batch):
                store.add(
                    documents=texts[i : i + batch],
                    embeddings=embeddings[i : i + batch],
                    ids=ids[i : i + batch],
                    metadatas=metas[i : i + batch],
                )

    # ── Web chunks ────────────────────────────────────────────────────────────
    if extra_chunks:
        web_texts:  list[str]  = []
        web_ids:    list[str]  = []
        web_metas:  list[dict] = []
        for chunk in (extra_chunks or []):
            text = chunk.get("text", "").strip()
            if not text:
                continue
            src  = str(chunk.get("source", "web"))
            page = chunk.get("page", "web")
            idx  = chunk.get("_idx", 0)
            web_texts.append(text)
            web_ids.append(_chunk_id(src, page, idx))
            web_metas.append({
                "source":   src,
                "category": str(chunk.get("category", "Web")),
                "page":     str(page),
                "snippet":  chunk.get("snippet", text[:120]),
            })

        if web_texts:
            embeddings = encode_passages(web_texts, batch_size=32)
            batch = 100
            for i in range(0, len(web_texts), batch):
                store.add(
                    documents=web_texts[i : i + batch],
                    embeddings=embeddings[i : i + batch],
                    ids=web_ids[i : i + batch],
                    metadatas=web_metas[i : i + batch],
                )

    return store
