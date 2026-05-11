"""
indexer.py -- ChromaDB index builder for the Streamlit app.

Fingerprint-based cache: if the set of PDFs has not changed since the
last build, the existing on-disk index is returned immediately.

Memory-safe: each PDF is processed and embedded independently so there
is never a large all-text list in memory at once.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import chromadb

from rag.embedder import encode_passages, EMBED_MODEL

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

_CHUNK_SIZE  = 800
_CHUNK_OVER  = 100
_CHROMA_DIR  = Path("/tmp/chroma_legal")
_FP_FILE     = _CHROMA_DIR / "fingerprint.txt"

# ── Fingerprint ────────────────────────────────────────────────────────────────

def _fingerprint(laws_dir: Path, extra_n: int = 0) -> str:
    parts: list[str] = [f"model:{EMBED_MODEL}"]   # invalidate cache on model change
    for cat_key in sorted(CATEGORIES):
        folder = laws_dir / cat_key
        if not folder.exists():
            continue
        for pdf in sorted(folder.glob("*.pdf")):
            parts.append(f"{cat_key}/{pdf.name}:{pdf.stat().st_size}")
    parts.append(f"web:{extra_n}")
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


def _try_load_existing(fp: str) -> chromadb.Collection | None:
    if not _FP_FILE.exists():
        return None
    if _FP_FILE.read_text().strip() != fp:
        return None
    try:
        client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
        col = client.get_collection("legal_docs")
        if col.count() > 0:
            return col
    except Exception:
        pass
    return None

# ── PDF extraction ─────────────────────────────────────────────────────────────

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
    raw = f"{source}|{page}|{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]

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
) -> chromadb.Collection:
    """
    Return a ChromaDB collection of all PDFs + optional web chunks.

    Memory-safe: each PDF is embedded and added to ChromaDB individually
    so peak memory is bounded by a single document, not the entire corpus.
    """
    extra_chunks = extra_chunks or []
    fp = _fingerprint(laws_dir, len(extra_chunks))

    cached = _try_load_existing(fp)
    if cached is not None:
        return cached

    # ── Full rebuild ──────────────────────────────────────────────────────────
    _CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
        client.delete_collection("legal_docs")
    except Exception:
        pass

    client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
    col = client.get_or_create_collection(
        "legal_docs",
        metadata={"hnsw:space": "cosine"},
    )

    # ── PDFs — one at a time to keep peak memory low ──────────────────────────
    for cat_key, cat_label in CATEGORIES.items():
        folder = laws_dir / cat_key
        if not folder.exists():
            continue
        for pdf_path in sorted(folder.glob("*.pdf")):
            source = pdf_path.stem
            texts: list[str] = []
            ids:   list[str] = []
            metas: list[dict] = []

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

            if not texts:
                continue

            embeddings = encode_passages(texts, batch_size=32)

            batch = 100
            for i in range(0, len(texts), batch):
                col.add(
                    documents=texts[i : i + batch],
                    embeddings=embeddings[i : i + batch],
                    ids=ids[i : i + batch],
                    metadatas=metas[i : i + batch],
                )
    # ── Web chunks ────────────────────────────────────────────────────────────
    if extra_chunks:
        web_texts:  list[str] = []
        web_ids:    list[str] = []
        web_metas:  list[dict] = []
        for chunk in extra_chunks:
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
                col.add(
                    documents=web_texts[i : i + batch],
                    embeddings=embeddings[i : i + batch],
                    ids=web_ids[i : i + batch],
                    metadatas=web_metas[i : i + batch],
                )

    _FP_FILE.write_text(fp)
    return col
