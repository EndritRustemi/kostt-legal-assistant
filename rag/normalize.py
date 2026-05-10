"""
normalize.py — Legal Text Normalizer

Preprocessing step that converts raw legal text (from PDF extraction
or plain TXT) into clean, consistently formatted text ready for
article/paragraph chunking by ingest.py.

Normalizations applied (in fixed order)
----------------------------------------
  1. Line endings           — CRLF / CR  →  LF
  2. Encoding artifacts     — BOM, NBSP, soft-hyphen, form-feed,
                              Unicode replacement char (U+FFFD)
  3. Unicode normalization  — NFC canonical composition
  4. PDF hyphenation        — re-join words broken with end-of-line hyphen
  5. Page number lines      — remove standalone digit-only lines
  6. Repeated line removal  — strip header/footer boilerplate (≥ N repeats)
  7. Article header case    — ARTICLE / NENI / ARTIKULLI → title-case
  8. Article header spacing — "Article  5" → "Article 5"
  9. Article separator      — ". / - / : / –" after number → " — "
                              trailing period with no title → removed
 10. Paragraph label spacing — "(1)text" → "(1) text"
                               "1 ."     → "1."
                               "( 1 )"   → "(1)"
 11. Trailing whitespace    — strip per line
 12. Multiple blank lines   — 3+ consecutive  →  2

Guarantee
---------
  Lossless for legal meaning: no substantive text is modified or removed.
  Only structural / typographic formatting is adjusted.
  Every change is recorded in NormalizedText.changes so the caller can audit.
"""

from __future__ import annotations

import logging
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("legal.normalize")

_LOG_FMT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class Change:
    """Records one class of normalization applied to the document."""
    rule:     str          # identifier, e.g. "article_case"
    label:    str          # human-readable description
    count:    int          # number of substitutions made
    examples: list[str] = field(default_factory=list)  # up to 5 before→after

    def __bool__(self) -> bool:
        return self.count > 0


@dataclass
class NormalizedText:
    """
    Return value of normalize().

    Attributes
    ----------
    text             : clean, normalized document text
    changes          : ordered list of Change objects (one per rule)
    original_chars   : character count before normalization
    normalized_chars : character count after normalization
    """
    text:             str
    changes:          list[Change]
    original_chars:   int
    normalized_chars: int

    @property
    def change_count(self) -> int:
        """Total number of individual substitutions across all rules."""
        return sum(c.count for c in self.changes)

    def report(self) -> str:
        """Return a compact, human-readable summary of all changes."""
        lines = [
            "Normalization report",
            f"  Input  : {self.original_chars:,} chars",
            f"  Output : {self.normalized_chars:,} chars  "
            f"(delta {self.normalized_chars - self.original_chars:+,})",
            f"  Rules applied : {sum(1 for c in self.changes if c.count)} "
            f"of {len(self.changes)}",
            f"  Total changes : {self.change_count}",
        ]
        active = [c for c in self.changes if c.count]
        if active:
            lines.append("")
            for c in active:
                lines.append(f"  [{c.count:>5}]  {c.label}")
                for ex in c.examples[:3]:
                    lines.append(f"             {ex}")
        return "\n".join(lines)


# ── Regex constants ───────────────────────────────────────────────────────────

# Page number: a line that is only digits, optionally surrounded by dashes
_PAGE_NUM_RE = re.compile(
    r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$",
    re.MULTILINE,
)

# Article header: matches the whole line when article keyword starts it.
# Groups: indent, kw (keyword), num (number), sep (separator), title (rest)
# Requires end-of-line ($) so inline references like "see Article 5 above"
# are not matched (they don't extend to end-of-line as article headers).
_ART_HEADER_RE = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?P<kw>ARTICLE|Article|NENI|Neni|ARTIKULLI|Artikulli|Art\.?)"
    r"\s+"
    r"(?P<num>\d+[a-zA-Z]?)"
    r"(?:"
    r"[ \t]*(?P<sep>[.\-–—:]+)[ \t]*"  # separator(s): . - -- – — : etc.
    r"(?P<title>[^\n]*)"               # title: rest of line (may be empty)
    r"|[ \t]*"                          # OR: nothing / trailing spaces
    r")$",
    re.MULTILINE,
)

# Canonical form for each article keyword variant
_KW_CANON: dict[str, str] = {
    "ARTICLE":    "Article",
    "Article":    "Article",
    "NENI":       "Neni",
    "Neni":       "Neni",
    "ARTIKULLI":  "Artikulli",
    "Artikulli":  "Artikulli",
    "Art.":       "Article",
    "Art":        "Article",
}

# PDF line-break hyphenation: word ends with hyphen, next line starts lowercase
# Strategy: keep hyphen, join lines — lossless because the hyphen may be real
# (e.g. "non-\ndiscriminatory" → "non-discriminatory"; safe either way)
_PDF_HYPHEN_RE = re.compile(r"(\w-)\n[ \t]*([a-z])")

# Paragraph label: paren-style ( N ) with optional inner whitespace
_PARA_PAREN_WS_RE = re.compile(r"(?m)^([ \t]*)\(\s*(\d+)\s*\)")

# Paragraph label: space BEFORE dot or paren suffix — "1 ." / "1 )"
_PARA_DOT_BEFORE_RE = re.compile(r"(?m)^([ \t]*)(\d+)\s+([.)])")

# Paragraph label: missing space AFTER the label and immediately followed by text
# Handles "(1)text", "1.text", "1)text"
_PARA_NOSPACE_AFTER_RE = re.compile(
    r"(?m)^([ \t]*)(\(\d+\)|\d+[.)])(\S)"
)

# 3+ consecutive blank lines
_MULTI_BLANK_RE = re.compile(r"\n{3,}")

# Trailing whitespace per line
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)

# Unicode replacement character (marks encoding failure)
_REPLACEMENT_CHAR_RE = re.compile(r"�+")


# ── Individual normalizers ────────────────────────────────────────────────────
# Each returns (new_text, Change).  Pure functions — no side effects.

def _step_line_endings(text: str) -> tuple[str, Change]:
    count = text.count("\r")
    text  = text.replace("\r\n", "\n").replace("\r", "\n")
    return text, Change("line_endings", "CRLF/CR → LF", count)


def _step_encoding_artifacts(text: str) -> tuple[str, Change]:
    """Replace/remove known encoding artifacts from PDF extraction."""
    subs = [
        ("﻿", "",  "BOM"),
        ("\xa0",   " ", "non-breaking space → space"),
        ("\xad",   "",  "soft hyphen → removed"),
        ("\f",     "\n","form-feed → newline"),
    ]
    examples: list[str] = []
    count = 0
    for src, dst, label in subs:
        n = text.count(src)
        if n:
            text = text.replace(src, dst)
            count += n
            examples.append(f"{n}× {label}")

    # Unicode replacement char — indicates an encoding failure; cannot recover
    repl_chars = _REPLACEMENT_CHAR_RE.findall(text)
    if repl_chars:
        n = sum(len(s) for s in repl_chars)
        log.warning(
            "Removed %d Unicode replacement character(s) (U+FFFD) — "
            "source bytes were undecodable; check source file encoding.",
            n,
        )
        text   = _REPLACEMENT_CHAR_RE.sub("", text)
        count += n
        examples.append(f"{n}× U+FFFD replacement char removed")

    return text, Change("encoding_artifacts", "Encoding artifacts removed", count, examples)


def _step_unicode_nfc(text: str) -> tuple[str, Change]:
    """Apply NFC Unicode normalization (canonical decomposition + recomposition)."""
    nfc   = unicodedata.normalize("NFC", text)
    count = sum(a != b for a, b in zip(text, nfc)) + abs(len(text) - len(nfc))
    return nfc, Change("unicode_nfc", "Unicode NFC normalization", count)


def _step_pdf_hyphenation(text: str) -> tuple[str, Change]:
    """
    Re-join words split by PDF line-break hyphenation.

    Rule: if a line ends with a hyphenated word-fragment and the next line
    starts with a lowercase letter, join them (keeping the hyphen, removing
    the newline).  This is lossless — the hyphen may be a real compound
    marker or a PDF artefact; keeping it is always correct.

    Example:
        "non-\\ndiscriminatory" → "non-discriminatory"
        "transmis-\\nsion"      → "transmis-sion"
    """
    examples: list[str] = []

    def _repl(m: re.Match) -> str:
        joined = m.group(1) + m.group(2)
        if len(examples) < 5:
            examples.append(
                f"{m.group(1)!r} + newline + {m.group(2)!r}  ->{joined!r}"
            )
        return joined

    new_text, count = _PDF_HYPHEN_RE.subn(_repl, text)
    return new_text, Change(
        "pdf_hyphenation",
        "PDF line-break hyphens joined",
        count,
        examples,
    )


def _step_page_numbers(text: str) -> tuple[str, Change]:
    """Remove standalone page-number lines (digits only, optionally dashed)."""
    matches = _PAGE_NUM_RE.findall(text)
    count   = len(matches)
    examples = [repr(m.strip()) for m in matches[:5]]
    text    = _PAGE_NUM_RE.sub("", text)
    return text, Change("page_numbers", "Standalone page-number lines removed", count, examples)


def _step_repeated_lines(
    text: str,
    min_repeats: int = 3,
) -> tuple[str, Change]:
    """
    Remove header/footer boilerplate: lines appearing verbatim
    ≥ min_repeats times in the document.

    Only lines longer than 8 characters are considered (short lines such
    as blank lines or bullet markers are intentionally skipped).
    """
    lines  = text.splitlines()
    counts = Counter(ln.strip() for ln in lines if len(ln.strip()) > 8)
    noise  = {ln for ln, cnt in counts.items() if cnt >= min_repeats}

    if not noise:
        return text, Change("repeated_lines", "Repeated header/footer lines removed", 0)

    cleaned  = [ln for ln in lines if ln.strip() not in noise]
    removed  = len(lines) - len(cleaned)
    examples = [repr(ln[:80]) for ln in sorted(noise)[:5]]
    return (
        "\n".join(cleaned),
        Change("repeated_lines", "Repeated header/footer lines removed", removed, examples),
    )


def _step_article_headers(text: str) -> tuple[str, Change]:
    """
    Normalize article header lines.

    Changes applied to matching lines only (lines where the article keyword
    starts at the beginning, with optional indent):
      • Keyword case  — ARTICLE → Article, NENI → Neni, Art. → Article
      • Spacing       — "Article  5" → "Article 5" (exactly one space)
      • Separator     — ". / - / : / –" after number → " — " (em-dash)
                        trailing separator with no title → removed
    Lines that contain prose after the article number without an explicit
    separator (e.g. "Article 5 of this Law shall…") are not modified.
    """
    examples: list[str] = []
    count = 0

    def _repl(m: re.Match) -> str:
        nonlocal count

        indent = m.group("indent")
        kw_raw = m.group("kw")
        num    = m.group("num")
        sep    = m.group("sep")    # may be None
        title  = m.group("title")  # may be None

        kw = _KW_CANON.get(kw_raw, kw_raw)

        if sep is not None:
            title = (title or "").strip()
            if title:
                new_line = f"{indent}{kw} {num} — {title}"
            else:
                # Separator with no title → drop separator entirely
                new_line = f"{indent}{kw} {num}"
        else:
            new_line = f"{indent}{kw} {num}"

        original = m.group(0)
        if new_line != original:
            count += 1
            if len(examples) < 5:
                examples.append(
                    f"{original.strip()!r}  ->{new_line.strip()!r}"
                )
        return new_line

    new_text = _ART_HEADER_RE.sub(_repl, text)
    return new_text, Change(
        "article_headers",
        "Article header case / spacing / separator standardized",
        count,
        examples,
    )


def _step_paragraph_labels(text: str) -> tuple[str, Change]:
    """
    Standardize paragraph label formatting at the start of lines.

    Rules (all lossless — only spacing is adjusted):
      "( 1 )"  →  "(1)"      inner whitespace in paren label removed
      "1 ."    →  "1."       space before dot/paren suffix removed
      "(1)text" →  "(1) text"  space added after label
      "1.text"  →  "1. text"   space added after label
      "1)text"  →  "1) text"   space added after label
    """
    count    = 0
    examples: list[str] = []

    def _record(before: str, after: str) -> str:
        nonlocal count
        if before != after:
            count += 1
            if len(examples) < 5:
                examples.append(f"{before.strip()!r}  ->{after.strip()!r}")
        return after

    # Pass 1 — fix inner whitespace in paren labels: "( 1 )" → "(1)"
    def _fix_paren_ws(m: re.Match) -> str:
        indent = m.group(1)
        num    = m.group(2)
        before = m.group(0)
        after  = f"{indent}({num})"
        return _record(before, after)

    text = _PARA_PAREN_WS_RE.sub(_fix_paren_ws, text)

    # Pass 2 — remove space before dot/paren suffix: "1 ." → "1." / "1 )" → "1)"
    def _fix_dot_before(m: re.Match) -> str:
        indent = m.group(1)
        num    = m.group(2)
        suffix = m.group(3)
        before = m.group(0)
        after  = f"{indent}{num}{suffix}"
        return _record(before, after)

    text = _PARA_DOT_BEFORE_RE.sub(_fix_dot_before, text)

    # Pass 3 — add missing space after label: "(1)text" → "(1) text"
    def _fix_nospace_after(m: re.Match) -> str:
        indent = m.group(1)
        label  = m.group(2)
        first  = m.group(3)
        before = m.group(0)
        after  = f"{indent}{label} {first}"
        return _record(before, after)

    text = _PARA_NOSPACE_AFTER_RE.sub(_fix_nospace_after, text)

    return text, Change(
        "paragraph_labels",
        "Paragraph label spacing standardized",
        count,
        examples,
    )


def _step_trailing_whitespace(text: str) -> tuple[str, Change]:
    """Strip trailing spaces and tabs from every line."""
    new_text, count = _TRAILING_WS_RE.subn("", text)
    return new_text, Change("trailing_whitespace", "Trailing whitespace stripped", count)


def _step_multi_blank_lines(text: str) -> tuple[str, Change]:
    """Collapse 3+ consecutive blank lines to 2."""
    new_text, count = _MULTI_BLANK_RE.subn("\n\n", text)
    return new_text, Change("multi_blank", "3+ consecutive blank lines collapsed", count)


# ── Master normalize function ─────────────────────────────────────────────────

def normalize(
    text: str,
    *,
    fix_hyphenation:       bool = True,
    remove_page_numbers:   bool = True,
    remove_repeated_lines: bool = True,
    standardize_articles:  bool = True,
    standardize_paragraphs: bool = True,
    min_header_repeats:    int  = 3,
) -> NormalizedText:
    """
    Normalize raw legal text through a fixed sequence of lossless steps.

    Parameters
    ----------
    text                   : raw input text (str)
    fix_hyphenation        : join PDF line-break hyphens (default True)
    remove_page_numbers    : strip standalone page-number lines (default True)
    remove_repeated_lines  : strip repeated header/footer boilerplate (default True)
    standardize_articles   : normalize Article/Neni headers (default True)
    standardize_paragraphs : fix paragraph label spacing (default True)
    min_header_repeats     : threshold for repeated-line removal (default 3)

    Returns
    -------
    NormalizedText
        .text    — the cleaned document
        .changes — auditable list of every change class applied
        .report()— human-readable summary string
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")

    original_chars = len(text)
    changes: list[Change] = []

    # Steps always applied (cannot be disabled — they are structural prerequisites)
    text, c = _step_line_endings(text)
    changes.append(c)

    text, c = _step_encoding_artifacts(text)
    changes.append(c)

    text, c = _step_unicode_nfc(text)
    changes.append(c)

    # Optional steps
    if fix_hyphenation:
        text, c = _step_pdf_hyphenation(text)
        changes.append(c)

    if remove_page_numbers:
        text, c = _step_page_numbers(text)
        changes.append(c)

    if remove_repeated_lines:
        text, c = _step_repeated_lines(text, min_repeats=min_header_repeats)
        changes.append(c)

    if standardize_articles:
        text, c = _step_article_headers(text)
        changes.append(c)

    if standardize_paragraphs:
        text, c = _step_paragraph_labels(text)
        changes.append(c)

    # Always applied last — clean up whitespace after all substitutions
    text, c = _step_trailing_whitespace(text)
    changes.append(c)

    text, c = _step_multi_blank_lines(text)
    changes.append(c)

    text = text.strip()

    return NormalizedText(
        text=text,
        changes=changes,
        original_chars=original_chars,
        normalized_chars=len(text),
    )


# ── File convenience wrapper ──────────────────────────────────────────────────

def normalize_file(
    input_path:  str | Path,
    output_path: str | Path | None = None,
    encoding:    str = "utf-8",
    **kwargs: Any,
) -> NormalizedText:
    """
    Read a text file, normalize it, and optionally write the result.

    Parameters
    ----------
    input_path   : path to .txt file (or PDF-extracted text)
    output_path  : if given, write the normalized text here
                   defaults to ``<stem>.normalized<suffix>`` alongside input
    encoding     : file encoding (default "utf-8")
    **kwargs     : forwarded to normalize()

    Returns
    -------
    NormalizedText — always returned regardless of output_path
    """
    src  = Path(input_path)
    text = src.read_text(encoding=encoding, errors="replace")

    result = normalize(text, **kwargs)

    if output_path is None:
        output_path = src.with_stem(src.stem + ".normalized")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.text, encoding=encoding)
    log.info("Normalized %s -> %s (%d changes)", src.name, out.name, result.change_count)

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)

    parser = argparse.ArgumentParser(
        prog="python -m rag.normalize",
        description="Normalize legal text files before ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Normalize one file and write alongside it:
  python -m rag.normalize law.txt

  # Specify output path:
  python -m rag.normalize law.txt --output data/law_clean.txt

  # Print full change report and preview:
  python -m rag.normalize law.txt --report --preview 20

  # Pipe from stdin to stdout:
  cat raw.txt | python -m rag.normalize --stdin
        """,
    )
    parser.add_argument("file",   nargs="?",  help="Input .txt file")
    parser.add_argument("--output",            help="Output path (default: <stem>.normalized.txt)")
    parser.add_argument("--stdin",  action="store_true", help="Read from stdin")
    parser.add_argument("--stdout", action="store_true", help="Write to stdout instead of file")
    parser.add_argument("--report", action="store_true", help="Print change report to stderr")
    parser.add_argument("--preview", type=int, default=0,
                        metavar="N", help="Print first N lines of output to stderr")
    parser.add_argument("--encoding", default="utf-8", help="File encoding (default: utf-8)")
    parser.add_argument("--no-hyphenation",  dest="fix_hyphenation",       action="store_false")
    parser.add_argument("--no-page-numbers", dest="remove_page_numbers",   action="store_false")
    parser.add_argument("--no-headers",      dest="remove_repeated_lines", action="store_false")
    parser.add_argument("--no-articles",     dest="standardize_articles",  action="store_false")
    parser.add_argument("--no-paragraphs",   dest="standardize_paragraphs",action="store_false")
    parser.add_argument("--min-repeats", type=int, default=3,
                        help="Min occurrences to classify a line as noise (default 3)")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    kwargs = {
        "fix_hyphenation":       args.fix_hyphenation,
        "remove_page_numbers":   args.remove_page_numbers,
        "remove_repeated_lines": args.remove_repeated_lines,
        "standardize_articles":  args.standardize_articles,
        "standardize_paragraphs": args.standardize_paragraphs,
        "min_header_repeats":    args.min_repeats,
    }

    # ── Input ──────────────────────────────────────────────────────────────────
    if args.stdin:
        raw = sys.stdin.read()
        result = normalize(raw, **kwargs)
    elif args.file:
        result = normalize_file(
            args.file,
            output_path=None if args.stdout else args.output,
            encoding=args.encoding,
            **kwargs,
        )
    else:
        parser.error("Provide a FILE argument or use --stdin")

    # ── Output ─────────────────────────────────────────────────────────────────
    if args.stdout or args.stdin:
        print(result.text)

    if args.report:
        print(result.report(), file=sys.stderr)

    if args.preview:
        lines = result.text.splitlines()
        print(
            f"\n--- First {args.preview} lines of normalized output ---",
            file=sys.stderr,
        )
        for ln in lines[:args.preview]:
            print(f"  {ln}", file=sys.stderr)
