"""
pdf_loader.py -- Legal PDF Text Extractor

Extracts clean, structured text from Kosovo legal PDFs.

Pipeline per document
---------------------
  1. Open PDF with PyMuPDF (primary) or pdfminer (fallback)
  2. Detect header/footer noise zones by finding text blocks that repeat
     in the top 8% or bottom 8% of a page across >= 3 pages
  3. Extract text blocks page-by-page, skipping detected noise blocks
  4. Post-process: fix encoding artifacts, re-join PDF line-break hyphens,
     strip standalone page numbers, collapse excessive blank lines
  5. Write one clean .txt file per PDF to the output directory

Article/Neni headings and paragraph breaks are preserved exactly.
No semantic normalization is applied -- run normalize.py as the next step.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("legal.pdf_loader")

_LOG_FMT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# -- Backend detection --------------------------------------------------------

try:
    import fitz as _fitz      # PyMuPDF
    _BACKEND = "pymupdf"
except ImportError:
    _fitz = None
    try:
        from pdfminer.high_level import extract_text as _pdfminer_extract  # type: ignore
        _BACKEND = "pdfminer"
    except ImportError:
        _pdfminer_extract = None
        _BACKEND = "none"

# -- Regex constants ----------------------------------------------------------

# Standalone page-number line: optional dashes around 1-4 digits
_PAGE_NUM_RE = re.compile(
    r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$",
    re.MULTILINE,
)

# PDF line-break hyphenation: word-fragment ending with hyphen + newline + lowercase
_HYPHEN_RE = re.compile(r"(\w-)\n[ \t]*([a-z])")

# Three or more consecutive blank lines -> two
_MULTI_BLANK_RE = re.compile(r"\n{3,}")

# Trailing whitespace on any line
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)

# -- Result type --------------------------------------------------------------

@dataclass
class ExtractionResult:
    """
    Outcome for a single PDF file.

    Attributes
    ----------
    pdf_path             : source PDF path
    txt_path             : output .txt path (may not exist if ok=False)
    page_count           : number of pages in the PDF (0 if unknown)
    char_count           : characters written to txt_path
    noise_blocks_removed : distinct header/footer block patterns removed
    warnings             : non-fatal notes (skipped, empty output, etc.)
    ok                   : False only on extraction failure
    """
    pdf_path:             Path
    txt_path:             Path
    page_count:           int
    char_count:           int
    noise_blocks_removed: int
    warnings:             list[str] = field(default_factory=list)
    ok:                   bool = True

    def __str__(self) -> str:
        status = "OK" if self.ok else "FAIL"
        return (
            f"[{status}] {self.pdf_path.name} -> {self.txt_path.name} "
            f"({self.page_count} pages, {self.char_count:,} chars, "
            f"{self.noise_blocks_removed} noise block(s) removed)"
        )


# -- PyMuPDF extraction -------------------------------------------------------

def _detect_noise_pymupdf(doc: Any) -> frozenset[str]:
    """
    Identify recurring header/footer text blocks using page geometry.

    A block is a noise candidate if:
      - its top edge is in the top 8% of the page, OR
      - its bottom edge is in the bottom 8% of the page
    AND its stripped text appears on at least `threshold` pages.

    Threshold: max(3, 15% of total pages) -- catches headers that appear
    on every page while avoiding false positives on short documents.
    """
    n_pages = len(doc)
    threshold = max(3, int(n_pages * 0.15))
    candidates: Counter[str] = Counter()

    for page in doc:
        h = page.rect.height
        if h == 0:
            continue
        seen_on_page: set[str] = set()
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, _bno, btype = block
            if btype != 0:          # skip image blocks
                continue
            stripped = text.strip()
            if len(stripped) < 4:   # ignore very short fragments
                continue
            y_top = y0 / h
            y_bot = y1 / h
            if y_top < 0.08 or y_bot > 0.92:
                # Count each unique text at most once per page
                if stripped not in seen_on_page:
                    candidates[stripped] += 1
                    seen_on_page.add(stripped)

    noise = frozenset(t for t, cnt in candidates.items() if cnt >= threshold)
    if noise:
        log.debug(
            "Noise patterns detected (%d): %s",
            len(noise),
            [n[:60] for n in noise],
        )
    return noise


def _extract_page_pymupdf(page: Any, noise: frozenset[str]) -> str:
    """
    Extract text from one page, skipping blocks whose content is in `noise`.
    Blocks are returned by PyMuPDF in reading order (top-to-bottom for
    single-column layouts typical of Kosovo legal documents).
    """
    parts: list[str] = []
    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text, _bno, btype = block
        if btype != 0:
            continue
        if text.strip() in noise:
            continue
        parts.append(text)
    return "".join(parts)


def _extract_pymupdf(path: Path) -> tuple[str, int, int]:
    """
    Extract all pages from `path` using PyMuPDF.

    Returns
    -------
    (raw_text, page_count, noise_block_count)
    """
    doc = _fitz.open(str(path))
    n_pages = len(doc)

    noise = _detect_noise_pymupdf(doc)

    page_texts: list[str] = []
    for page in doc:
        pt = _extract_page_pymupdf(page, noise)
        stripped = pt.strip()
        if stripped:
            page_texts.append(stripped)

    doc.close()

    # Join pages with a blank line so paragraph breaks are visible at boundaries
    raw = "\n\n".join(page_texts)
    return raw, n_pages, len(noise)


# -- pdfminer fallback --------------------------------------------------------

def _extract_pdfminer(path: Path) -> tuple[str, int, int]:
    """
    Extract text using pdfminer. No block-level noise filtering is possible;
    repeated-line removal in _clean() handles headers/footers instead.
    """
    text = _pdfminer_extract(str(path)) or ""
    return text, 0, 0


# -- Post-extraction cleanup --------------------------------------------------

def _remove_repeated_lines(text: str, min_repeats: int = 3) -> tuple[str, int]:
    """
    Remove verbatim lines that appear >= min_repeats times in the document.
    Used as a backstop after block-level noise filtering.
    Only considers lines longer than 8 characters to avoid stripping
    legitimate short lines such as section numbers.
    Returns (cleaned_text, lines_removed).
    """
    lines  = text.splitlines()
    counts = Counter(ln.strip() for ln in lines if len(ln.strip()) > 8)
    noise  = {ln for ln, cnt in counts.items() if cnt >= min_repeats}
    if not noise:
        return text, 0
    cleaned = [ln for ln in lines if ln.strip() not in noise]
    removed = len(lines) - len(cleaned)
    return "\n".join(cleaned), removed


def _clean(raw: str, min_header_repeats: int = 3) -> str:
    """
    Apply post-extraction cleanup to text from any backend.

    Steps (all lossless for legal meaning):
      1. Normalize line endings
      2. Remove PDF control characters (NBSP, soft-hyphen, form-feed, BOM)
      3. Unicode NFC normalization
      4. Re-join PDF line-break hyphens
      5. Remove standalone page-number lines
      6. Remove repeated header/footer lines (backstop)
      7. Strip trailing whitespace per line
      8. Collapse 3+ consecutive blank lines to 2
    """
    # 1. Line endings
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    # 2. PDF control characters
    text = (
        text
        .replace("﻿", "")   # BOM
        .replace("\xa0",   " ")  # NBSP
        .replace("\xad",   "")   # soft hyphen
        .replace("\f",     "\n") # form-feed
    )

    # 3. Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # 4. PDF hyphenation (keep hyphen, drop newline)
    text = _HYPHEN_RE.sub(lambda m: m.group(1) + m.group(2), text)

    # 5. Page number lines
    text = _PAGE_NUM_RE.sub("", text)

    # 6. Repeated header/footer lines
    text, removed = _remove_repeated_lines(text, min_header_repeats)
    if removed:
        log.debug("Repeated-line pass removed %d line(s)", removed)

    # 7. Trailing whitespace
    text = _TRAILING_WS_RE.sub("", text)

    # 8. Collapse blank lines
    text = _MULTI_BLANK_RE.sub("\n\n", text)

    return text.strip()


# -- Single-file entry point --------------------------------------------------

def extract_pdf(
    pdf_path:           str | Path,
    output_dir:         str | Path = "pdf_text",
    overwrite:          bool       = False,
    encoding:           str        = "utf-8",
    min_header_repeats: int        = 3,
) -> ExtractionResult:
    """
    Extract text from a single PDF and write a .txt file.

    Parameters
    ----------
    pdf_path           : path to the source PDF
    output_dir         : directory for the .txt output (created if absent)
    overwrite          : if False, skip files that already have a .txt
    encoding           : output file encoding (default utf-8)
    min_header_repeats : repeated-line threshold for the fallback pass

    Returns
    -------
    ExtractionResult
    """
    pdf_path   = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / (pdf_path.stem + ".txt")
    warnings: list[str] = []

    if not pdf_path.exists():
        msg = f"file not found: {pdf_path}"
        log.error(msg)
        return ExtractionResult(
            pdf_path=pdf_path, txt_path=txt_path,
            page_count=0, char_count=0, noise_blocks_removed=0,
            warnings=[msg], ok=False,
        )

    if txt_path.exists() and not overwrite:
        log.info("Skipping %s (already extracted; use overwrite=True to force)",
                 pdf_path.name)
        return ExtractionResult(
            pdf_path=pdf_path, txt_path=txt_path,
            page_count=0,
            char_count=len(txt_path.read_text(encoding=encoding, errors="replace")),
            noise_blocks_removed=0,
            warnings=["skipped: output already exists"],
        )

    # -- Extraction -----------------------------------------------------------
    if _BACKEND == "pymupdf":
        try:
            raw, n_pages, noise_count = _extract_pymupdf(pdf_path)
        except Exception as exc:
            msg = f"PyMuPDF extraction failed: {exc}"
            log.error("%s: %s", pdf_path.name, msg)
            return ExtractionResult(
                pdf_path=pdf_path, txt_path=txt_path,
                page_count=0, char_count=0, noise_blocks_removed=0,
                warnings=[msg], ok=False,
            )

    elif _BACKEND == "pdfminer":
        try:
            raw, n_pages, noise_count = _extract_pdfminer(pdf_path)
        except Exception as exc:
            msg = f"pdfminer extraction failed: {exc}"
            log.error("%s: %s", pdf_path.name, msg)
            return ExtractionResult(
                pdf_path=pdf_path, txt_path=txt_path,
                page_count=0, char_count=0, noise_blocks_removed=0,
                warnings=[msg], ok=False,
            )

    else:
        msg = (
            "No PDF backend available. "
            "Install PyMuPDF:  pip install pymupdf   (preferred)\n"
            "            or:  pip install pdfminer.six"
        )
        log.error(msg)
        return ExtractionResult(
            pdf_path=pdf_path, txt_path=txt_path,
            page_count=0, char_count=0, noise_blocks_removed=0,
            warnings=[msg], ok=False,
        )

    # -- Cleanup + write ------------------------------------------------------
    text = _clean(raw, min_header_repeats=min_header_repeats)

    if not text:
        msg = "extracted text is empty after cleaning -- PDF may be scanned/image-only"
        log.warning("%s: %s", pdf_path.name, msg)
        warnings.append(msg)

    txt_path.write_text(text, encoding=encoding)

    log.info(
        "Extracted %s -> %s | pages=%d chars=%d noise=%d",
        pdf_path.name, txt_path.name, n_pages, len(text), noise_count,
    )

    return ExtractionResult(
        pdf_path=pdf_path,
        txt_path=txt_path,
        page_count=n_pages,
        char_count=len(text),
        noise_blocks_removed=noise_count,
        warnings=warnings,
    )


# -- Folder entry point -------------------------------------------------------

def extract_folder(
    input_dir:          str | Path = "pdfs_raw",
    output_dir:         str | Path = "pdf_text",
    overwrite:          bool       = False,
    encoding:           str        = "utf-8",
    pattern:            str        = "*.pdf",
    min_header_repeats: int        = 3,
) -> list[ExtractionResult]:
    """
    Extract all PDFs in `input_dir` and write .txt files to `output_dir`.

    Parameters
    ----------
    input_dir          : folder containing the source PDFs (must exist)
    output_dir         : folder for output .txt files (created if absent)
    overwrite          : re-extract files that already have a .txt
    encoding           : output encoding (default utf-8)
    pattern            : glob pattern for PDF files (default "*.pdf")
    min_header_repeats : repeated-line threshold forwarded to extract_pdf

    Returns
    -------
    list[ExtractionResult]
        One result per PDF found, in alphabetical order.
        Inspect .ok and .warnings on each result for per-file status.

    Raises
    ------
    FileNotFoundError
        If input_dir does not exist.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    pdfs = sorted(input_dir.glob(pattern))
    if not pdfs:
        log.warning("No PDF files matching %r found in %s", pattern, input_dir)
        return []

    log.info(
        "Found %d PDF(s) in %s  [backend=%s]",
        len(pdfs), input_dir, _BACKEND,
    )

    results: list[ExtractionResult] = []
    for pdf in pdfs:
        result = extract_pdf(
            pdf, output_dir,
            overwrite=overwrite,
            encoding=encoding,
            min_header_repeats=min_header_repeats,
        )
        results.append(result)

    ok_count   = sum(1 for r in results if r.ok and not any("skipped" in w for w in r.warnings))
    skip_count = sum(1 for r in results if any("skipped" in w for w in r.warnings))
    fail_count = sum(1 for r in results if not r.ok)

    log.info(
        "Batch complete: %d extracted, %d skipped, %d failed",
        ok_count, skip_count, fail_count,
    )
    return results


# -- CLI ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)

    parser = argparse.ArgumentParser(
        prog="python -m rag.pdf_loader",
        description="Extract clean text from legal PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
backend : {_BACKEND}

examples:
  # Process entire pdfs_raw/ folder -> pdf_text/
  python -m rag.pdf_loader

  # Custom paths
  python -m rag.pdf_loader --input docs/pdfs --output docs/txt

  # Single file
  python -m rag.pdf_loader --file law.pdf --output pdf_text

  # Force re-extraction of already-processed files
  python -m rag.pdf_loader --overwrite
        """,
    )
    parser.add_argument(
        "--input",  "-i",
        default="pdfs_raw",
        metavar="DIR",
        help="Folder containing source PDFs (default: pdfs_raw)",
    )
    parser.add_argument(
        "--output", "-o",
        default="pdf_text",
        metavar="DIR",
        help="Folder for output .txt files (default: pdf_text)",
    )
    parser.add_argument(
        "--file", "-f",
        metavar="PDF",
        help="Process a single PDF file instead of a folder",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract files that already have a .txt output",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Output file encoding (default: utf-8)",
    )
    parser.add_argument(
        "--min-repeats",
        type=int,
        default=3,
        metavar="N",
        help="Lines repeated >= N times are treated as header/footer noise (default: 3)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if _BACKEND == "none":
        print(
            "ERROR: no PDF backend installed.\n"
            "  pip install pymupdf        (preferred)\n"
            "  pip install pdfminer.six   (fallback)",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.file:
        results = [extract_pdf(
            args.file,
            output_dir=args.output,
            overwrite=args.overwrite,
            encoding=args.encoding,
            min_header_repeats=args.min_repeats,
        )]
    else:
        try:
            results = extract_folder(
                input_dir=args.input,
                output_dir=args.output,
                overwrite=args.overwrite,
                encoding=args.encoding,
                min_header_repeats=args.min_repeats,
            )
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(2)

    # -- Print summary --------------------------------------------------------
    sep = "-" * 64
    print()
    print(sep)
    for r in results:
        print(f"  {r}")
        for w in r.warnings:
            print(f"      WARNING: {w}")
    print(sep)
    total_chars = sum(r.char_count for r in results)
    fail_count  = sum(1 for r in results if not r.ok)
    print(f"  Files processed : {len(results)}")
    print(f"  Total output    : {total_chars:,} chars")
    print(f"  Failures        : {fail_count}")
    print(sep)

    sys.exit(1 if fail_count else 0)
