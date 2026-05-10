"""
ingest.py — Legal Document Ingestor

Converts raw legal documents (TXT or PDF) into a structured
processed_chunks.json ready for the citation engine.

Output schema per chunk
-----------------------
  id              : str  — "doc-slug_artN_parN"
  text            : str  — cleaned legal provision text
  source          : dict
      document_title : str
      institution    : str
      document_type  : str
  article         : str  — "Article N [— Title]"   (top-level, for retrieve.py / cite.py)
  paragraph       : str  — "Paragraph N"            (top-level, for cite.py)
  legal_reference : dict
      article   : str
      paragraph : str
  semantic        : dict
      keywords : list[str]  — 5–10 legal terms
      summary  : str        — 1-sentence summary

Strict rules enforced
---------------------
  - Every chunk carries both article and paragraph references
  - No merging of multiple articles into one chunk
  - PDF formatting artifacts are removed; legal meaning is preserved exactly
  - Chunks without recoverable legal text are silently skipped with a log warning
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests  # already in requirements.txt

from .normalize import normalize as _normalize_text

# ── Soft dependency: pymupdf / fitz (PDF extraction) ─────────────────────────

try:
    import fitz as _fitz   # pymupdf — in requirements.txt
    _FITZ = True
except ImportError:
    _FITZ = False

# ── Logging ───────────────────────────────────────────────────────────────────

_LOG_FMT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"
log = logging.getLogger("legal.ingest")

# ── Gemini ────────────────────────────────────────────────────────────────────

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-2.5-flash-lite:generateContent"
)

# ── Article header patterns ───────────────────────────────────────────────────
# Matches: "Article 5", "Neni 5", "Artikulli 5a", "ARTICLE 12", "Art. 3" etc.
# Groups: kw (keyword), num (article number), title (optional same-line title)
_ARTICLE_RE = re.compile(
    r"^\s*"
    r"(?P<kw>ARTICLE|Article|Neni|NENI|Artikulli|ARTIKULLI|Art\.)\s+"
    r"(?P<num>\d+[a-zA-Z]?)"
    r"(?:[ \t]*[.\-–—:])?"   # optional separator: . - – — :
    r"[ \t]*(?P<title>[^\n]*)",         # optional title: rest of line
    re.MULTILINE,
)

# ── Paragraph start patterns within an article body ───────────────────────────
# Handles: "(1) text", "1. text", "1) text"
_PARA_RE = re.compile(
    r"(?m)^[ \t]*"
    r"(?:"
    r"\((?P<p1>\d+)\)"   # (1)
    r"|(?P<p2>\d+)\."    # 1.
    r"|(?P<p3>\d+)\)"    # 1)
    r")\s+",
)

# ── Albanian + English legal stop words for keyword extraction ────────────────

_STOP: frozenset[str] = frozenset({
    # English
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "on", "at",
    "by", "with", "shall", "must", "may", "will", "is", "are", "be",
    "been", "being", "has", "have", "had", "do", "does", "did", "not",
    "no", "any", "all", "each", "every", "their", "its", "this", "that",
    "these", "those", "as", "such", "within", "which", "who", "where",
    "when", "if", "but", "however", "provided", "pursuant", "under",
    "above", "below", "following", "applicable", "said", "same", "other",
    "further", "also", "only", "including", "according", "regarding",
    "concerning", "between", "among", "into", "from", "through", "during",
    "before", "after", "without", "unless", "until", "whether", "both",
    "either", "neither", "than", "thereof", "thereto", "therein", "hereby",
    "hereunder", "respective", "herein", "relevant", "upon", "related",
    # Albanian
    "dhe", "ose", "i", "e", "te", "ne", "me", "per", "nga", "si",
    "qe", "por", "nese", "gjithashtu", "sipas", "se", "ku", "kur",
    "jo", "po", "nje", "cdo", "gjithe", "ai", "ajo", "ata", "ato",
    "kjo", "ky", "kete", "kesaj", "atij", "asaj", "atyre", "rast",
    "menyre", "pa", "edhe", "vetem", "kohe", "do", "ka", "jane",
    "eshte", "kishte", "kishin", "behet", "bejne", "ben", "duhet",
    "mund", "ndaj", "mbi", "nen", "nder", "cfare", "kush",
})

# ── Document type + institution auto-detection ────────────────────────────────

_DOCTYPE_PATS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:Law|Ligj)\s+No\.?",  re.I), "Law"),
    (re.compile(r"\bNetwork\s+Code\b",       re.I), "Network Code"),
    (re.compile(r"\bMarket\s+Rules?\b",      re.I), "Market Rules"),
    (re.compile(r"\bGrid\s+Code\b",          re.I), "Grid Code"),
    (re.compile(r"\bRegulation\b",           re.I), "Regulation"),
    (re.compile(r"\bDirective\b",            re.I), "Directive"),
    (re.compile(r"\bDecision\b",             re.I), "Decision"),
    (re.compile(r"\bCode\b",                 re.I), "Code"),
    (re.compile(r"\bAgreement\b",            re.I), "Agreement"),
    (re.compile(r"\bContract\b",             re.I), "Contract"),
]

_INSTITUTION_PATS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bAssembly\s+of\s+Kosovo\b",       re.I), "Assembly of Kosovo"),
    (re.compile(r"\bKuvendi\s+i\s+Kosov",            re.I), "Kuvendi i Kosovës"),
    (re.compile(r"\bZRRE\b",                          re.I), "ZRRE (Energy Regulatory Office)"),
    (re.compile(r"\bEnergy\s+Regulatory\s+Office\b", re.I), "Energy Regulatory Office"),
    (re.compile(r"\bKOSTT\b",                         re.I), "KOSTT"),
    (re.compile(r"\bGovernment\s+of\s+Kosovo\b",      re.I), "Government of Kosovo"),
    (re.compile(r"\bQeveria\s+e\s+Kosov",             re.I), "Qeveria e Kosovës"),
]



# ── ID and slug utilities ─────────────────────────────────────────────────────

def _slugify(text: str, max_len: int = 28) -> str:
    """Produce a short, readable ASCII slug."""
    nfkd      = unicodedata.normalize("NFKD", text)
    ascii_str = nfkd.encode("ascii", "ignore").decode()
    slug      = re.sub(r"[^a-z0-9]+", "-", ascii_str.lower()).strip("-")
    slug      = re.sub(r"-+", "-", slug)
    return slug[:max_len].strip("-")


def _make_id(doc_title: str, art_num: str, par_num: int | str) -> str:
    """Generate a unique, readable chunk ID in format doc-slug_artN_parN."""
    doc_slug = _slugify(doc_title, max_len=24)
    return f"{doc_slug}_art{art_num.lower()}_par{par_num}"


# ── Article splitting ─────────────────────────────────────────────────────────

@dataclass
class _RawArticle:
    keyword: str   # "Article" / "Neni"
    num:     str   # "5" or "5a"
    title:   str   # optional article title (may be empty string)
    body:    str   # full article text after the header line


def _split_articles(text: str) -> list[_RawArticle]:
    """
    Split document text into individual articles.
    Text before the first article header (preamble/recitals) is discarded.
    """
    matches = list(_ARTICLE_RE.finditer(text))
    if not matches:
        log.warning(
            "No article headers found — confirm the document uses "
            "'Article N' or 'Neni N' headings"
        )
        return []

    articles: list[_RawArticle] = []
    for i, m in enumerate(matches):
        body_start = m.end()
        body_end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body       = text[body_start:body_end].strip()

        # Title: strip noise characters from the same-line title group
        title = re.sub(
            r"^[-–—:\s]+|[-–—:\s]+$",
            "",
            (m.group("title") or "").strip(),
        )

        # If title still empty, try the first body line as the title:
        # must be short, not start with a digit, not look like an article or paragraph.
        if not title and body:
            first = body.splitlines()[0].strip()
            if (
                len(first) < 90
                and first
                and not first[0].isdigit()
                and not _PARA_RE.match(first)
                and not _ARTICLE_RE.match(first)
            ):
                title = first
                body  = "\n".join(body.splitlines()[1:]).strip()

        articles.append(_RawArticle(
            keyword=m.group("kw"),
            num=m.group("num"),
            title=title,
            body=body,
        ))

    log.debug("Found %d article(s)", len(articles))
    return articles


# ── Paragraph splitting ───────────────────────────────────────────────────────

@dataclass
class _RawParagraph:
    num:  int
    text: str


def _split_paragraphs(body: str) -> list[_RawParagraph]:
    """
    Split article body into numbered paragraphs.

    Strategy (in priority order):
      1. Explicit numeric markers: "(1)", "1.", "1)"
      2. Blank-line separated blocks (each >= 60 chars)
      3. Whole body as paragraph 1
    """
    matches = list(_PARA_RE.finditer(body))

    if matches:
        paras: list[_RawParagraph] = []
        for i, m in enumerate(matches):
            num_str    = m.group("p1") or m.group("p2") or m.group("p3")
            text_start = m.end()
            text_end   = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            text       = body[text_start:text_end].strip()
            if text:
                paras.append(_RawParagraph(num=int(num_str), text=text))
        if paras:
            return paras

    # Fallback 1: blank-line-separated blocks of substance
    blocks = [b.strip() for b in re.split(r"\n\s*\n", body) if len(b.strip()) >= 60]
    if len(blocks) > 1:
        return [_RawParagraph(num=i + 1, text=b) for i, b in enumerate(blocks)]

    # Fallback 2: whole article body as a single paragraph
    if body.strip():
        return [_RawParagraph(num=1, text=body.strip())]

    return []


# ── Semantic extraction ───────────────────────────────────────────────────────

def _first_sentence(text: str) -> str:
    """Return first complete sentence (ends with . ; ? !), or first 120 chars."""
    m = re.search(r"[^.;?!\n]{20,}[.;?!]", text.strip())
    return m.group().strip() if m else text[:120].strip()


def _heuristic_keywords(text: str, n: int = 8) -> list[str]:
    """
    Extract top-n meaningful legal terms via frequency analysis.
    Capitalized multi-word phrases (e.g. "Transmission System Operator")
    are weighted 3x over single-word tokens.
    """
    # Multi-word capitalized phrases
    phrase_raw    = re.findall(
        r"\b[A-ZÀ-ɏ][a-zÀ-ɏ]+"
        r"(?:\s+[A-ZÀ-ɏ][a-zÀ-ɏ]+)+",
        text,
    )
    phrase_counts = Counter(p.lower() for p in phrase_raw if len(p) > 5)

    # Individual tokens (min 4 chars, excluding stop words)
    tokens       = re.findall(r"\b\w{4,}\b", text.lower())
    token_counts = Counter(t for t in tokens if t not in _STOP)

    combined: Counter[str] = Counter()
    for phrase, cnt in phrase_counts.items():
        combined[phrase] += cnt * 3
    for tok, cnt in token_counts.items():
        combined[tok] += cnt

    # Deduplicate: skip a single-word token already covered by a selected phrase
    selected: list[str] = []
    for kw, _ in combined.most_common(n * 4):
        if not any(kw != sel and kw in sel for sel in selected):
            selected.append(kw)
        if len(selected) >= n:
            break

    return selected


def _semantic_heuristic(text: str) -> dict[str, Any]:
    return {
        "keywords": _heuristic_keywords(text),
        "summary":  _first_sentence(text),
    }


def _semantic_gemini(text: str, api_key: str) -> dict[str, Any]:
    """
    Use Gemini to produce high-quality keywords and a 1-sentence summary.
    Falls back to heuristic on any error (never raises).
    """
    prompt = (
        "You are a legal terminology expert specializing in Kosovo energy law.\n"
        "Given the legal provision below, extract:\n"
        "  1. 5-10 key legal terms or concepts (specific nouns/phrases, not generic words)\n"
        "  2. A precise 1-sentence summary of what this provision establishes or requires\n\n"
        "Return ONLY valid JSON — no markdown, no code fences, no prose:\n"
        '{"keywords": ["term1", "term2", ...], "summary": "One sentence."}\n\n'
        f"LEGAL PROVISION:\n{text}"
    )
    try:
        resp = requests.post(
            f"{_GEMINI_URL}?key={api_key}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 300},
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        raw = re.sub(r"^```(?:json)?\s*|\s*```\s*$", "", raw.strip())
        result = json.loads(raw)
        if (
            isinstance(result.get("keywords"), list)
            and isinstance(result.get("summary"), str)
            and result["keywords"]
            and result["summary"]
        ):
            return result
        log.warning("Gemini returned unexpected semantic structure; falling back to heuristic")
    except Exception as exc:
        log.warning("Gemini semantic call failed (%s); falling back to heuristic", exc)

    return _semantic_heuristic(text)


# ── Metadata auto-detection ───────────────────────────────────────────────────

def _detect_metadata(header: str) -> dict[str, str]:
    """
    Infer document_type and institution from the first ~30 lines.
    Returns empty strings for fields that cannot be detected.
    """
    doc_type    = next((v for p, v in _DOCTYPE_PATS    if p.search(header)), "")
    institution = next((v for p, v in _INSTITUTION_PATS if p.search(header)), "")
    return {"document_type": doc_type, "institution": institution}


# ── PDF text extraction ───────────────────────────────────────────────────────

def _extract_pdf(path: Path) -> str:
    """Extract plain text from a PDF using pymupdf (fitz)."""
    if not _FITZ:
        raise ImportError(
            "pymupdf is required for PDF extraction but is not installed.\n"
            "Install with:  pip install pymupdf"
        )
    doc   = _fitz.open(str(path))
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(pages)


# ── Main ingestor class ───────────────────────────────────────────────────────

class LegalIngestor:
    """
    Convert a legal document (text or PDF) into structured chunks.

    Parameters
    ----------
    document_title : str
        Human-readable document name used verbatim in citations.
    institution    : str
        Issuing body. Pass "" to auto-detect from the document header.
    document_type  : str
        E.g. "Law", "Regulation", "Network Code". Pass "" to auto-detect.
    api_key        : str
        Gemini API key. If empty, heuristic extraction is used (no API calls).
    enrich         : bool
        When True (and api_key set) Gemini is called once per chunk for
        keywords/summary. Produces higher quality but is slower for large docs.
    rate_limit_s   : float
        Minimum seconds between Gemini calls to respect quota limits (default 1.0).
    """

    def __init__(
        self,
        document_title: str,
        institution:    str   = "",
        document_type:  str   = "",
        api_key:        str   = "",
        enrich:         bool  = True,
        rate_limit_s:   float = 1.0,
    ) -> None:
        if not document_title.strip():
            raise ValueError("document_title must not be empty")
        self.document_title = document_title.strip()
        self._institution   = institution.strip()
        self._document_type = document_type.strip()
        self._api_key       = api_key.strip()
        self._enrich        = enrich and bool(self._api_key)
        self._rate_limit    = rate_limit_s
        self._last_call_t   = 0.0

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_source(self, detected: dict[str, str]) -> dict[str, str]:
        return {
            "document_title": self.document_title,
            "institution":    self._institution or detected.get("institution", ""),
            "document_type":  self._document_type or detected.get("document_type", ""),
        }

    def _get_semantic(self, text: str) -> dict[str, Any]:
        if not self._enrich:
            return _semantic_heuristic(text)
        elapsed = time.monotonic() - self._last_call_t
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        result = _semantic_gemini(text, self._api_key)
        self._last_call_t = time.monotonic()
        return result

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest_text(self, text: str) -> list[dict[str, Any]]:
        """
        Process a raw legal text string and return a list of chunk dicts.
        Preamble text before the first article header is discarded.
        """
        norm     = _normalize_text(text)
        if norm.change_count:
            log.debug("normalize: %d change(s) applied", norm.change_count)
        cleaned  = norm.text
        header   = "\n".join(cleaned.splitlines()[:30])
        detected = _detect_metadata(header)
        source   = self._build_source(detected)

        articles = _split_articles(cleaned)
        if not articles:
            return []

        chunks:  list[dict[str, Any]] = []
        skipped: int = 0

        for art in articles:
            base_label  = f"{art.keyword} {art.num}"
            art_display = f"{base_label} — {art.title}" if art.title else base_label

            paras = _split_paragraphs(art.body)
            if not paras:
                log.debug("Skipping empty article %s in '%s'",
                          base_label, self.document_title)
                skipped += 1
                continue

            for para in paras:
                if not para.text.strip():
                    continue

                chunk_id  = _make_id(self.document_title, art.num, para.num)
                par_label = f"Paragraph {para.num}"

                chunks.append({
                    "id":       chunk_id,
                    "text":     para.text,
                    # Nested source (full metadata)
                    "source":   source,
                    # Top-level fields required by retrieve.py (_is_legal check)
                    # and used by cite.py for inline citation formatting
                    "article":  art_display,
                    "paragraph": par_label,
                    # Rich nested reference for downstream analytics / filtering
                    "legal_reference": {
                        "article":   art_display,
                        "paragraph": par_label,
                    },
                    "semantic": self._get_semantic(para.text),
                })

        log.info(
            "Ingested '%s' -> %d chunk(s), %d empty article(s) skipped",
            self.document_title, len(chunks), skipped,
        )
        return chunks

    def ingest_file(self, path: str | Path) -> list[dict[str, Any]]:
        """
        Read a .txt or .pdf file and return chunks.

        Raises
        ------
        FileNotFoundError  — file does not exist
        ValueError         — unsupported file extension
        ImportError        — PDF requested but pymupdf not installed
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            log.info("Extracting text from PDF: %s", path.name)
            text = _extract_pdf(path)
        elif suffix in {".txt", ".md", ""}:
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            raise ValueError(
                f"Unsupported extension {suffix!r}. Supported: .txt, .md, .pdf"
            )

        return self.ingest_text(text)


# ── Convenience functions ─────────────────────────────────────────────────────

def ingest_file(
    path:           str | Path,
    document_title: str,
    institution:    str  = "",
    document_type:  str  = "",
    api_key:        str  = "",
    enrich:         bool = False,
) -> list[dict[str, Any]]:
    """
    One-shot helper: ingest a single file and return chunks.
    ``enrich=False`` by default for fast, zero-API-call processing.
    """
    return LegalIngestor(
        document_title=document_title,
        institution=institution,
        document_type=document_type,
        api_key=api_key,
        enrich=enrich,
    ).ingest_file(path)


def save_chunks(
    chunks:      list[dict[str, Any]],
    output_path: str | Path = "processed_chunks.json",
) -> Path:
    """
    Write chunks to a JSON file (UTF-8, 2-space indent).
    Creates parent directories if they do not exist.
    Returns the resolved output path.
    """
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Saved %d chunk(s) -> %s", len(chunks), out)
    return out


def load_chunks(path: str | Path = "processed_chunks.json") -> list[dict[str, Any]]:
    """Load and return chunks from a JSON file produced by save_chunks()."""
    data = Path(path).read_text(encoding="utf-8")
    return json.loads(data)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)

    parser = argparse.ArgumentParser(
        prog="python -m rag.ingest",
        description="Convert legal documents into processed_chunks.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Fast — heuristic semantics, no API key needed:
  python -m rag.ingest law.pdf \\
      --title "Law No. 05/L-085 on Electricity" \\
      --institution "Assembly of Kosovo" --type Law

  # Rich — Gemini semantics (one API call per chunk):
  python -m rag.ingest network_code.txt \\
      --title "ZRRE Network Code" --institution ZRRE --type "Network Code" \\
      --enrich --api-key $GEMINI_API_KEY

  # Append multiple files into one corpus:
  python -m rag.ingest law.pdf code.txt \\
      --title "KOSTT Grid Code" --output data/corpus.json --append
        """,
    )
    parser.add_argument("files", nargs="+", metavar="FILE",
                        help=".txt or .pdf file(s) to ingest")
    parser.add_argument("--title",       required=True,
                        help="Document title (used verbatim in citations)")
    parser.add_argument("--institution", default="",
                        help="Issuing institution (auto-detected if omitted)")
    parser.add_argument("--type",        default="", dest="doc_type",
                        help="Document type: Law, Regulation, Network Code, etc.")
    parser.add_argument("--output",      default="processed_chunks.json",
                        help="Output JSON path (default: processed_chunks.json)")
    parser.add_argument("--enrich",      action="store_true",
                        help="Use Gemini for keyword/summary extraction (slower)")
    parser.add_argument("--api-key",
                        default=os.environ.get("GEMINI_API_KEY", ""),
                        dest="api_key",
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--append",      action="store_true",
                        help="Append new chunks to an existing output file")
    parser.add_argument("--debug",       action="store_true",
                        help="Enable DEBUG logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.enrich and not args.api_key:
        print("ERROR: --enrich requires --api-key or GEMINI_API_KEY env var",
              file=sys.stderr)
        sys.exit(1)

    all_chunks: list[dict] = []
    if args.append and Path(args.output).exists():
        all_chunks = load_chunks(args.output)
        print(f"Loaded {len(all_chunks)} existing chunk(s) from {args.output}")

    ingestor = LegalIngestor(
        document_title=args.title,
        institution=args.institution,
        document_type=args.doc_type,
        api_key=args.api_key,
        enrich=args.enrich,
    )

    err_count = 0
    for file_path in args.files:
        try:
            chunks = ingestor.ingest_file(file_path)
            print(f"  {Path(file_path).name}: {len(chunks)} chunk(s)")
            all_chunks.extend(chunks)
        except Exception as exc:
            print(f"  ERROR {Path(file_path).name}: {exc}", file=sys.stderr)
            err_count += 1

    if all_chunks:
        out = save_chunks(all_chunks, args.output)
        print(f"\nTotal: {len(all_chunks)} chunk(s) -> {out}")
    else:
        print("\nNo chunks produced — check input files and document format.",
              file=sys.stderr)

    if err_count:
        sys.exit(1)
