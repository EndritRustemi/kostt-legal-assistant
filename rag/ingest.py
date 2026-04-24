"""
PDF ingestion: lexo → copëzo → embedo (lokal) → ruaj në ChromaDB
"""

import time
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # mbështet shqipen, falas, lokal

CATEGORIES = {
    "kosove":      "Legjislacioni i Kosovës",
    "zrre":        "Rregulloret ZRRE",
    "entso-e":     "Rregulloret ENTSO-E",
    "eu":          "Direktivat EU",
    "strategjike": "Dokumentet Strategjike",
    "vendime":     "Vendime",
    "te-tjera":    "Të Tjera",
}


def _extract_pages(pdf_path: Path) -> list[dict]:
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def _chunk(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 60]


def build_index(laws_dir: Path, api_key: str = "") -> chromadb.Collection:
    model = SentenceTransformer(EMBED_MODEL)

    chroma = chromadb.Client()
    col = chroma.get_or_create_collection(
        name="kostt_legal",
        metadata={"hnsw:space": "cosine"},
    )

    doc_id = 0
    for cat_folder, cat_label in CATEGORIES.items():
        folder = laws_dir / cat_folder
        if not folder.exists():
            continue
        for pdf in folder.glob("*.pdf"):
            pages = _extract_pages(pdf)
            all_chunks = []
            all_meta = []
            for page_data in pages:
                for chunk in _chunk(page_data["text"]):
                    all_chunks.append(chunk)
                    all_meta.append({
                        "source": pdf.name,
                        "category": cat_label,
                        "page": page_data["page"],
                        "snippet": chunk[:120],
                    })

            if not all_chunks:
                continue

            embeddings = model.encode(all_chunks, show_progress_bar=False).tolist()
            ids = [str(doc_id + i) for i in range(len(all_chunks))]
            col.add(
                ids=ids,
                embeddings=embeddings,
                documents=all_chunks,
                metadatas=all_meta,
            )
            doc_id += len(all_chunks)

    return col


def count_pdfs(laws_dir: Path) -> dict[str, int]:
    counts = {}
    for cat_folder, cat_label in CATEGORIES.items():
        folder = laws_dir / cat_folder
        counts[cat_label] = len(list(folder.glob("*.pdf"))) if folder.exists() else 0
    return counts
