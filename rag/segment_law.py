"""
segment_law.py -- Legal Text Segmenter

Converts extracted .txt files (from pdf_loader.py) into structured JSON
chunks ready for embedding and retrieval. One chunk = one paragraph under
one article. No chunk may exist without both an article and paragraph
reference.

Pipeline per file
-----------------
  1. Load .txt file and apply OCR-noise cleaning
  2. Auto-detect document_title, institution, document_type from content
  3. Split into articles using Neni / Article / Artikulli headers
  4. Split each article body into numbered paragraphs
  5. Assign unique IDs and emit flat chunk dicts
  6. Write per-file JSON to chunks/ and accumulate combined corpus

Output schema (per chunk)
--------------------------
  {
    "id":             "<doc-slug>_art<N>_par<N>",
    "text":           "<clean paragraph text>",
    "article":        "Neni 5 -- Qellimi i ligjit",
    "paragraph":      "Paragraph 1",
    "document_title": "Law No. 05/L-085 on Electricity",
    "institution":    "Assembly of Kosovo",
    "document_type":  "Law"
  }

Guarantees
----------
  - Every chunk carries a non-empty article and paragraph reference (C3/C4)
  - Articles are never merged across chunk boundaries
  - Preamble text before the first article header is discarded
  - OCR ligatures, curly quotes, and control characters are normalised
  - Legal meaning is preserved exactly
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("legal.segment_law")

_LOG_FMT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


# -- OCR noise -----------------------------------------------------------------

# Unicode ligatures produced by some PDF extractors
_OCR_LIGATURES: dict[str, str] = {
    "ﬀ": "ff",   # ﬀ
    "ﬁ": "fi",   # ﬁ
    "ﬂ": "fl",   # ﬂ
    "ﬃ": "ffi",  # ﬃ
    "ﬄ": "ffl",  # ﬄ
    "ﬅ": "st",   # ﬅ  (rare)
    "ﬆ": "st",   # ﬆ  (rare)
}

# Curly / typographic quotes -> straight (legal text uses straight quotes)
_OCR_QUOTES: dict[str, str] = {
    "‘": "'",  # left single
    "’": "'",  # right single
    "‚": "'",  # single low-9
    "‛": "'",  # single high-reversed-9
    "“": '"',  # left double
    "”": '"',  # right double
    "„": '"',  # double low-9
    "‟": '"',  # double high-reversed-9
    "«": '"',  # <<
    "»": '"',  # >>
    "‹": "'",  # < (single guillemet)
    "›": "'",  # > (single guillemet)
}

# Combined translation table (applied in one pass)
_OCR_TABLE = str.maketrans({**_OCR_LIGATURES, **_OCR_QUOTES})

# Control / invisible characters removed wholesale (soft-hyphen, NBSP, BOM, FF)
_CTRL_RE = re.compile(r"[\xad​‌‍﻿]")

# Multiple spaces within a single line -> single space
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")

# Line that contains only punctuation / noise (dashes, dots, underscores, digits alone)
_NOISE_LINE_RE = re.compile(
    r"^[ \t]*"
    r"(?:"
    r"[-–—_=~]{3,}"  # 3+ dashes / underscores / tildes
    r"|\.{3,}"                  # 3+ dots (ellipsis artifacts)
    r"|\*{3,}"                  # 3+ asterisks
    r")[ \t]*$",
    re.MULTILINE,
)

# Standalone page-number line: 1-4 digits, optionally with dashes around it
_PAGE_NUM_RE = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$",
                          re.MULTILINE)


# -- Article / paragraph regexes -----------------------------------------------

# Article header: supports Albanian (Neni/Artikulli) and English (Article)
# in any capitalisation; optional separator and same-line title.
# Requires BOL (with optional indent) so inline references ("sipas nenit 5...")
# are not matched -- they appear mid-line, not at line start.
_ARTICLE_RE = re.compile(
    r"^\s*"
    r"(?P<kw>ARTICLE|Article|NENI|Neni|ARTIKULLI|Artikulli|Art\.?)"
    r"\s+"
    r"(?P<num>\d+[a-zA-Z]?)"
    r"(?:[ \t]*[-.–—:]+)?"   # optional separator(s)
    r"[ \t]*(?P<title>[^\n]*)",
    re.MULTILINE,
)

# Paragraph start markers within an article body
_PARA_RE = re.compile(
    r"(?m)^[ \t]*"
    r"(?:"
    r"\((?P<p1>\d+)\)"   # (1)
    r"|(?P<p2>\d+)\."    # 1.
    r"|(?P<p3>\d+)\)"    # 1)
    r")\s+",
)

# -- Metadata auto-detection ---------------------------------------------------

_DOCTYPE_PATS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:Law|Ligji?)\s+[Nn]o\.?",   re.I), "Law"),
    (re.compile(r"\bNetwork\s+Code\b",              re.I), "Network Code"),
    (re.compile(r"\bGrid\s+Code\b",                 re.I), "Grid Code"),
    (re.compile(r"\bMarket\s+Rules?\b",             re.I), "Market Rules"),
    (re.compile(r"\bRregull(?:ore|im)\b",           re.I), "Regulation"),
    (re.compile(r"\bRegulation\b",                  re.I), "Regulation"),
    (re.compile(r"\bDirective\b",                   re.I), "Directive"),
    (re.compile(r"\bDecision\b",                    re.I), "Decision"),
    (re.compile(r"\bVendim\b",                      re.I), "Decision"),
    (re.compile(r"\bCode\b",                        re.I), "Code"),
    (re.compile(r"\bAgreement\b",                   re.I), "Agreement"),
    (re.compile(r"\bContract\b",                    re.I), "Contract"),
]

_INSTITUTION_PATS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bAssembly\s+of\s+Kosovo\b",      re.I), "Assembly of Kosovo"),
    (re.compile(r"\bKuvendi\s+i\s+Kosov",           re.I), "Kuvendi i Kosoves"),
    (re.compile(r"\bZRRE\b",                        re.I), "ZRRE (Energy Regulatory Office)"),
    (re.compile(r"\bEnergy\s+Regulatory\s+Office\b",re.I), "Energy Regulatory Office"),
    (re.compile(r"\bKOSTT\b",                       re.I), "KOSTT"),
    (re.compile(r"\bGovernment\s+of\s+Kosovo\b",    re.I), "Government of Kosovo"),
    (re.compile(r"\bQeveria\s+e\s+Kosov",           re.I), "Qeveria e Kosoves"),
    (re.compile(r"\bMinistry\s+of\b",               re.I), "Ministry of Kosovo"),
]


# -- Internal data model -------------------------------------------------------

@dataclass
class _Article:
    keyword: str   # "Neni" / "Article"
    num:     str   # "5" or "5a"
    title:   str   # optional same-line title (may be empty)
    body:    str   # text between this header and the next


@dataclass
class _Paragraph:
    num:  int
    text: str


# -- Helpers -------------------------------------------------------------------

def _slugify(text: str, max_len: int = 24) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    slug = re.sub(r"[^a-z0-9]+", "-", nfkd.encode("ascii", "ignore").decode().lower())
    return re.sub(r"-+", "-", slug).strip("-")[:max_len].strip("-")


def _make_id(doc_title: str, art_num: str, par_num: int | str) -> str:
    return f"{_slugify(doc_title)}_art{art_num.lower()}_par{par_num}"


def _clean_ocr(text: str) -> str:
    """
    Remove OCR / PDF extraction artifacts. Lossless for legal meaning.

    Fixes applied:
      - CRLF and bare CR -> LF
      - BOM, soft hyphen, zero-width chars -> removed
      - NBSP, form-feed, vertical tab -> space / newline
      - Unicode ligatures (ﬁ, ﬀ, ...) -> ASCII equivalents
      - Curly quotes -> straight quotes
      - NFC unicode normalization
      - Noise-only lines (3+ dashes, dots, asterisks) -> removed
      - Standalone page-number lines -> removed
      - Multiple consecutive spaces within a line -> single space
      - Trailing whitespace per line -> stripped
    """
    # Line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Control characters
    text = text.replace("\xa0", " ")   # NBSP -> space
    text = text.replace("\f",   "\n")  # form-feed -> newline
    text = text.replace("\v",   "\n")  # vertical tab -> newline
    text = _CTRL_RE.sub("", text)      # soft-hyphen, zero-width, BOM

    # Ligatures and quotes
    text = text.translate(_OCR_TABLE)

    # NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Noise lines
    text = _NOISE_LINE_RE.sub("", text)

    # Page numbers
    text = _PAGE_NUM_RE.sub("", text)

    # Multiple spaces within lines, trailing whitespace
    lines = [_MULTI_SPACE_RE.sub(" ", ln).rstrip() for ln in text.splitlines()]
    text  = "\n".join(lines)

    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _detect_metadata(text: str, filename_stem: str) -> dict[str, str]:
    """
    Infer document_title, institution, and document_type from the first
    40 lines of the document. Falls back to the filename stem if the title
    cannot be found.
    """
    header = "\n".join(text.splitlines()[:40])

    doc_type    = next((v for p, v in _DOCTYPE_PATS    if p.search(header)), "")
    institution = next((v for p, v in _INSTITUTION_PATS if p.search(header)), "")

    # Title heuristic: first non-empty line that is not an article header,
    # not a paragraph marker, and not longer than 160 chars.
    title = ""
    for ln in text.splitlines():
        stripped = ln.strip()
        if (
            stripped
            and len(stripped) <= 160
            and not _ARTICLE_RE.match(stripped)
            and not _PARA_RE.match(stripped)
            and not _PAGE_NUM_RE.match(stripped)
        ):
            title = stripped
            break

    if not title:
        # Fall back to filename, convert slug to title-case
        title = filename_stem.replace("-", " ").replace("_", " ").title()

    return {
        "document_title": title,
        "institution":    institution,
        "document_type":  doc_type,
    }


def _split_articles(text: str) -> list[_Article]:
    """
    Split document text into articles using Neni / Article / Artikulli headers.
    Preamble text before the first header is discarded.
    Returns an empty list (with a logged warning) if no headers are found.
    """
    matches = list(_ARTICLE_RE.finditer(text))
    if not matches:
        log.warning(
            "No article headers found -- confirm the document uses "
            "'Neni N' or 'Article N' headings"
        )
        return []

    articles: list[_Article] = []
    for i, m in enumerate(matches):
        body_start = m.end()
        body_end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body       = text[body_start:body_end].strip()

        # Clean noise from the same-line title group
        raw_title = (m.group("title") or "").strip()
        title = re.sub(r"^[-.–—:\s]+|[-.–—:\s]+$", "", raw_title)

        # If title is still empty, try the first body line as the article title.
        # Must be short, not a digit, not look like a new article or paragraph.
        if not title and body:
            first = body.splitlines()[0].strip()
            if (
                first
                and len(first) < 100
                and not first[0].isdigit()
                and not _PARA_RE.match(first)
                and not _ARTICLE_RE.match(first)
            ):
                title = first
                body  = "\n".join(body.splitlines()[1:]).strip()

        articles.append(_Article(
            keyword=m.group("kw"),
            num=m.group("num"),
            title=title,
            body=body,
        ))

    log.debug("Found %d article(s)", len(articles))
    return articles


def _split_paragraphs(body: str) -> list[_Paragraph]:
    """
    Split article body into numbered paragraphs.

    Strategy (in priority order):
      1. Explicit numeric markers: (N), N., N)
      2. Blank-line separated blocks of >= 50 characters each
      3. Whole body as Paragraph 1 (guaranteed non-empty result)
    """
    matches = list(_PARA_RE.finditer(body))

    if matches:
        paras: list[_Paragraph] = []
        for i, m in enumerate(matches):
            num_str    = m.group("p1") or m.group("p2") or m.group("p3")
            text_start = m.end()
            text_end   = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            text       = body[text_start:text_end].strip()
            if text:
                paras.append(_Paragraph(num=int(num_str), text=text))
        if paras:
            return paras

    # Fallback 1: blank-line separated blocks of substance
    blocks = [b.strip() for b in re.split(r"\n\s*\n", body) if len(b.strip()) >= 50]
    if len(blocks) > 1:
        return [_Paragraph(num=i + 1, text=b) for i, b in enumerate(blocks)]

    # Fallback 2: whole body is Paragraph 1
    if body.strip():
        return [_Paragraph(num=1, text=body.strip())]

    return []


# -- Public API ----------------------------------------------------------------

def segment_text(
    text:           str,
    document_title: str,
    institution:    str = "",
    document_type:  str = "",
) -> list[dict[str, Any]]:
    """
    Segment a legal text string into structured chunks.

    Parameters
    ----------
    text           : raw or pre-cleaned legal document text
    document_title : document name used verbatim in every chunk and in IDs
    institution    : issuing body (used as-is; auto-detected if empty)
    document_type  : e.g. "Law", "Regulation", "Code" (auto-detected if empty)

    Returns
    -------
    list[dict]
        One dict per paragraph, with keys:
        id, text, article, paragraph, document_title, institution, document_type.
        Empty if no article headers are found.

    Raises
    ------
    ValueError
        If document_title is empty (cannot generate valid IDs without it).
    """
    if not document_title.strip():
        raise ValueError("document_title must not be empty")

    cleaned  = _clean_ocr(text)
    articles = _split_articles(cleaned)

    chunks:  list[dict[str, Any]] = []
    skipped: int = 0

    for art in articles:
        base_label  = f"{art.keyword} {art.num}"
        art_display = f"{base_label} -- {art.title}" if art.title else base_label

        paras = _split_paragraphs(art.body)
        if not paras:
            log.warning(
                "Article %s has no recoverable paragraph text -- skipped",
                base_label,
            )
            skipped += 1
            continue

        for para in paras:
            chunks.append({
                "id":             _make_id(document_title, art.num, para.num),
                "text":           para.text,
                "article":        art_display,
                "paragraph":      f"Paragraph {para.num}",
                "document_title": document_title,
                "institution":    institution,
                "document_type":  document_type,
            })

    if skipped:
        log.warning(
            "%d article(s) skipped due to empty body in '%s'",
            skipped, document_title,
        )

    log.info(
        "Segmented '%s' -> %d chunk(s) from %d article(s)",
        document_title, len(chunks), len(articles),
    )
    return chunks


def segment_file(
    txt_path:       str | Path,
    output_path:    str | Path | None = None,
    document_title: str = "",
    institution:    str = "",
    document_type:  str = "",
    encoding:       str = "utf-8",
) -> list[dict[str, Any]]:
    """
    Segment a single .txt file and optionally write its chunks to a JSON file.

    Parameters
    ----------
    txt_path       : path to a .txt file (output of pdf_loader.py)
    output_path    : if given, write chunks to this JSON file
    document_title : overrides auto-detection from document content
    institution    : overrides auto-detection
    document_type  : overrides auto-detection
    encoding       : file encoding (default utf-8)

    Returns
    -------
    list[dict]
        Chunks produced from this file.

    Raises
    ------
    FileNotFoundError  -- if txt_path does not exist
    """
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Input file not found: {txt_path}")

    text = txt_path.read_text(encoding=encoding, errors="replace")

    # Auto-detect metadata from content; caller overrides take precedence
    detected = _detect_metadata(text, txt_path.stem)
    title   = document_title.strip() or detected["document_title"]
    inst    = institution.strip()    or detected["institution"]
    doctype = document_type.strip()  or detected["document_type"]

    chunks = segment_text(text, title, inst, doctype)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2),
            encoding=encoding,
        )
        log.info("Wrote %d chunk(s) -> %s", len(chunks), out)

    return chunks


def segment_folder(
    input_dir:   str | Path = "pdf_text",
    output_dir:  str | Path = "chunks",
    combined:    str | Path = "chunks/processed_chunks.json",
    encoding:    str        = "utf-8",
    overwrite:   bool       = False,
    pattern:     str        = "*.txt",
) -> list[Path]:
    """
    Segment all .txt files in input_dir and write one JSON file per source
    document plus a combined corpus file.

    Parameters
    ----------
    input_dir  : directory containing .txt files from pdf_loader.py
    output_dir : directory for per-file chunk JSON files
    combined   : path to the combined processed_chunks.json
    encoding   : file encoding (default utf-8)
    overwrite  : re-segment files that already have a JSON output
    pattern    : glob pattern for input files (default "*.txt")

    Returns
    -------
    list[Path]
        Paths to each per-file JSON that was written (in alphabetical order).
        Does NOT include the combined file path.

    Raises
    ------
    FileNotFoundError  -- if input_dir does not exist
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    combined   = Path(combined)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    txt_files = sorted(input_dir.glob(pattern))
    if not txt_files:
        log.warning("No files matching %r found in %s", pattern, input_dir)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    combined.parent.mkdir(parents=True, exist_ok=True)

    log.info("Segmenting %d file(s) from %s", len(txt_files), input_dir)

    written:      list[Path]       = []
    all_chunks:   list[dict]       = []
    skipped_count: int             = 0
    error_count:   int             = 0

    for txt in txt_files:
        out_json = output_dir / (txt.stem + ".json")

        if out_json.exists() and not overwrite:
            log.info("Skipping %s (already segmented; use overwrite=True to force)",
                     txt.name)
            # Still load existing chunks into the combined corpus
            try:
                existing = json.loads(out_json.read_text(encoding=encoding))
                all_chunks.extend(existing)
            except Exception:
                pass
            skipped_count += 1
            continue

        try:
            chunks = segment_file(
                txt, output_path=out_json, encoding=encoding,
            )
        except Exception as exc:
            log.error("Failed to segment %s: %s", txt.name, exc)
            error_count += 1
            continue

        all_chunks.extend(chunks)
        written.append(out_json)

    # Write combined corpus
    combined.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding=encoding,
    )
    log.info(
        "Combined corpus: %d chunk(s) -> %s",
        len(all_chunks), combined,
    )

    ok_count = len(written)
    log.info(
        "Batch complete: %d segmented, %d skipped, %d failed",
        ok_count, skipped_count, error_count,
    )

    return written


# -- CLI -----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)

    parser = argparse.ArgumentParser(
        prog="python -m rag.segment_law",
        description="Segment legal .txt files into structured JSON chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Segment entire pdf_text/ folder -> chunks/
  python -m rag.segment_law

  # Custom paths
  python -m rag.segment_law --input pdf_text --output chunks

  # Single file with explicit metadata
  python -m rag.segment_law --file law.txt \\
      --title "Law No. 05/L-085 on Electricity" \\
      --institution "Assembly of Kosovo" --type Law

  # Re-segment already-processed files
  python -m rag.segment_law --overwrite

  # Preview without writing files
  python -m rag.segment_law --file law.txt --dry-run
        """,
    )
    parser.add_argument("--input",   "-i", default="pdf_text", metavar="DIR",
                        help="Folder with .txt files (default: pdf_text)")
    parser.add_argument("--output",  "-o", default="chunks",   metavar="DIR",
                        help="Folder for per-file JSON output (default: chunks)")
    parser.add_argument("--combined",       default="chunks/processed_chunks.json",
                        metavar="FILE",
                        help="Combined corpus JSON path (default: chunks/processed_chunks.json)")
    parser.add_argument("--file",    "-f", metavar="TXT",
                        help="Process a single .txt file instead of a folder")
    parser.add_argument("--title",          default="", metavar="TITLE",
                        help="Document title (overrides auto-detection)")
    parser.add_argument("--institution",    default="", metavar="INST",
                        help="Issuing institution (overrides auto-detection)")
    parser.add_argument("--type",           default="", dest="doc_type", metavar="TYPE",
                        help="Document type: Law, Regulation, Code, etc.")
    parser.add_argument("--overwrite",      action="store_true",
                        help="Re-segment files that already have JSON output")
    parser.add_argument("--dry-run",        action="store_true",
                        help="Print chunks to stdout; do not write any files")
    parser.add_argument("--debug",          action="store_true",
                        help="Enable DEBUG logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Single file mode ──────────────────────────────────────────────────────
    if args.file:
        try:
            chunks = segment_file(
                args.file,
                output_path=None if args.dry_run else (
                    Path(args.output) / (Path(args.file).stem + ".json")
                ),
                document_title=args.title,
                institution=args.institution,
                document_type=args.doc_type,
            )
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(2)

        if args.dry_run:
            print(json.dumps(chunks, ensure_ascii=False, indent=2))
        else:
            sep = "-" * 60
            print(sep)
            for c in chunks:
                print(f"  [{c['id']}]")
                print(f"    {c['article']}  |  {c['paragraph']}")
                print(f"    {c['text'][:80]}{'...' if len(c['text']) > 80 else ''}")
            print(sep)
            print(f"  Chunks : {len(chunks)}")
            print(sep)
        sys.exit(0)

    # ── Folder mode ───────────────────────────────────────────────────────────
    try:
        written = segment_folder(
            input_dir=args.input,
            output_dir=args.output,
            combined=args.combined,
            overwrite=args.overwrite,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    sep = "-" * 60
    print()
    print(sep)
    for p in written:
        print(f"  {p}")
    print(sep)
    print(f"  Files written : {len(written)}")
    print(f"  Combined      : {args.combined}")
    print(sep)
    sys.exit(0)
