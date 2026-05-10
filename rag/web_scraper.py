"""
web_scraper.py -- Web page scraper for trusted legal sources.

Fetches a URL, extracts clean readable text (removes nav/footer/scripts),
splits it into chunks, and returns them in the same format as PDF chunks
so they can be added to the ChromaDB index without any special handling.

Public API
----------
    scrape_url(url, label, category)  -> list[dict]
    load_web_sources(path)            -> list[dict]
    save_web_sources(sources, path)   -> None

Chunk format (compatible with retriever.py)
-------------------------------------------
    {
      "text":     str,
      "source":   str,   # URL
      "category": str,   # e.g. "ZRRE", "KOSTT", "ENTSO-E"
      "page":     "web",
      "snippet":  str,   # first 120 chars
      "_idx":     int,   # internal, used for stable ID generation
    }
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import requests

# BeautifulSoup is an optional dependency; we degrade gracefully.
try:
    from bs4 import BeautifulSoup, Tag
    _BS4 = True
except ImportError:
    _BS4 = False

# ── Config ─────────────────────────────────────────────────────────────────────

_CHUNK_SIZE   = 500    # target characters per chunk
_CHUNK_OVER   = 60     # overlap
_REQUEST_TIMEOUT = 20  # seconds
_MIN_CHUNK_LEN   = 80  # discard very short fragments

# HTML elements whose content is almost always noise
_NOISE_TAGS = {
    "script", "style", "noscript", "nav", "footer", "header",
    "aside", "form", "button", "select", "option", "iframe",
    "svg", "img", "figure", "figcaption",
}

# Pre-built list of known Kosovo / EU energy sector sources
KNOWN_SOURCES: list[dict[str, str]] = [
    {"url": "https://www.kostt.com",           "label": "KOSTT",    "category": "KOSTT"},
    {"url": "https://www.zrre.rks-gov.net",    "label": "ZRRE",     "category": "ZRRE"},
    {"url": "https://www.kek-energy.com",      "label": "KEK",      "category": "KEK"},
    {"url": "https://www.kesco.com",           "label": "KESCO",    "category": "KESCO"},
    {"url": "https://www.keds-energy.com",     "label": "KEDS",     "category": "KEDS"},
    {"url": "https://www.entsoe.eu",           "label": "ENTSO-E",  "category": "ENTSO-E"},
    {"url": "https://www.energy-community.org","label": "Komuniteti i Energjisë", "category": "Energy Community"},
    {"url": "https://gzk.rks-gov.net",        "label": "Gazeta Zyrtare e Kosovës", "category": "Gazeta Zyrtare"},
]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _extract_text_bs4(html: str) -> str:
    """Extract visible text using BeautifulSoup, removing noise elements."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove noise tags entirely
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()

    # Try to find the main content area first
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id=re.compile(r"content|main|body", re.I))
        or soup.find(class_=re.compile(r"content|main|body|article", re.I))
        or soup.body
        or soup
    )

    text = main.get_text(separator="\n") if main else soup.get_text(separator="\n")
    return text


def _extract_text_regex(html: str) -> str:
    """Fallback extractor when bs4 is not installed."""
    # Strip tags
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>",  " ", text,  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;",  "&", text)
    text = re.sub(r"&lt;",   "<", text)
    text = re.sub(r"&gt;",   ">", text)
    return text


def _clean(text: str) -> str:
    """Normalise whitespace and remove near-empty lines."""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) > 20:          # skip very short lines (menu items etc.)
            lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split(text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVER) -> list[str]:
    """Split on paragraph breaks first, then by size."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) + 2 <= size:
            buf = (buf + "\n\n" + para).strip() if buf else para
        else:
            if buf:
                chunks.append(buf)
            if len(para) <= size:
                buf = para
            else:
                # Hard split long paragraph
                start = 0
                while start < len(para):
                    end = min(start + size, len(para))
                    if end < len(para):
                        snap = para.rfind(" ", start, end)
                        if snap > start:
                            end = snap
                    chunks.append(para[start:end].strip())
                    start = end - overlap
                buf = ""
    if buf:
        chunks.append(buf)
    return [c for c in chunks if len(c) >= _MIN_CHUNK_LEN]


# ── Public API ─────────────────────────────────────────────────────────────────

def scrape_url(
    url: str,
    label: str = "",
    category: str = "Web",
) -> list[dict[str, Any]]:
    """
    Fetch a URL and return a list of text chunks ready for indexing.

    Parameters
    ----------
    url      : the page to fetch (must be HTTP/HTTPS)
    label    : human-readable name shown in source cards (defaults to URL)
    category : source category shown in the UI (e.g. "ZRRE", "KOSTT")

    Returns
    -------
    list[dict]  — empty list on any fetch/parse error (errors are not raised)
    """
    label = label or url

    try:
        resp = requests.get(
            url,
            timeout=_REQUEST_TIMEOUT,
            headers={"User-Agent": "KOSTT-Legal-RAG/1.0 (legal research bot)"},
        )
        resp.raise_for_status()
    except Exception:
        return []

    html = resp.text

    if _BS4:
        raw = _extract_text_bs4(html)
    else:
        raw = _extract_text_regex(html)

    clean = _clean(raw)
    if not clean:
        return []

    chunks = _split(clean)
    return [
        {
            "text":     chunk,
            "source":   label,
            "category": category,
            "page":     "web",
            "snippet":  chunk[:120],
            "_idx":     i,
        }
        for i, chunk in enumerate(chunks)
    ]


def load_web_sources(path: str | Path) -> list[dict[str, str]]:
    """
    Load the persisted list of web sources from a JSON file.

    Returns [] if the file doesn't exist or is invalid.
    Each entry: {"url": str, "label": str, "category": str}
    """
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return [
            s for s in data
            if isinstance(s, dict) and s.get("url")
        ]
    except Exception:
        return []


def save_web_sources(sources: list[dict[str, str]], path: str | Path) -> None:
    """Persist the web sources list to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(sources, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
