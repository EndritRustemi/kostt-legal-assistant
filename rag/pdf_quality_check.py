"""
pdf_quality_check.py -- Legal Text Quality Validator

Validates the quality of text extracted from Kosovo legal PDFs.
Designed to run on the output of pdf_loader.py (the pdf_text/ folder).

Checks performed
----------------
  QC-1  Article detection   -- Were articles found? Is numbering gapless?
  QC-2  Paragraph integrity -- Are paragraphs well-formed and complete?
  QC-3  OCR quality         -- Is the character distribution plausible?
  QC-4  Duplicate sections  -- Are article/paragraph bodies repeated?

Scoring
-------
  Each check contributes to a 0-100 quality score.
  Score >= 70 : VALID   (safe to ingest)
  Score <  70 : INVALID (needs manual review or re-extraction)

  The score is computed as:
      100 - sum(all_penalties), clamped to [0, 100]

  Penalty weights are tuned so that a single critical failure (no articles,
  completely garbled text) produces a score below 70 by itself.

Output
------
  QualityReport   -- per-file result (score, valid, errors, warnings, stats)
  FolderReport    -- batch result over a directory

Public API
----------
  check_text(text, file_name="")  -> QualityReport
  check_file(path)                -> QualityReport
  check_folder(folder, pattern)   -> FolderReport
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from hashlib import md5
from pathlib import Path
from typing import Any

from .law_parser_utils import (
    find_articles,
    find_paragraphs,
    extract_structure,
    ArticleMatch,
)

_LOG_FMT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

INVALID_THRESHOLD = 70   # score below this -> INVALID


# -- Issue model --------------------------------------------------------------

@dataclass
class Issue:
    """One quality problem found in a document."""
    severity: str    # "error" | "warning"
    code:     str    # e.g. "QC1_NO_ARTICLES"
    message:  str
    detail:   str = ""
    penalty:  int = 0   # score deduction for this issue

    def __str__(self) -> str:
        tag = "ERROR" if self.severity == "error" else "WARN "
        line = f"  [{tag}] [{self.code}] {self.message}"
        if self.detail:
            line += f"\n           {self.detail}"
        return line


# -- Quality report -----------------------------------------------------------

@dataclass
class QualityReport:
    """
    Quality result for a single text file.

    Attributes
    ----------
    file_name : source file name (empty when checking raw text)
    score     : 0-100 quality score
    valid     : True iff score >= INVALID_THRESHOLD (70)
    errors    : list of Issue with severity="error"
    warnings  : list of Issue with severity="warning"
    stats     : dict of raw measurements used to compute the score
    """
    file_name: str
    score:     int
    valid:     bool
    errors:    list[Issue]
    warnings:  list[Issue]
    stats:     dict[str, Any]

    @property
    def issues(self) -> list[Issue]:
        """All issues (errors first, then warnings)."""
        return self.errors + self.warnings

    def summary(self) -> str:
        sep = "-" * 64
        verdict = "VALID" if self.valid else "INVALID"
        lines = [
            sep,
            f"  File    : {self.file_name or '<raw text>'}",
            f"  Score   : {self.score}/100",
            f"  Result  : {verdict}",
            f"  Errors  : {len(self.errors)}",
            f"  Warnings: {len(self.warnings)}",
        ]
        # Key stats
        s = self.stats
        lines += [
            f"  Articles     : {s.get('article_count', 0)}",
            f"  Paragraphs   : {s.get('paragraph_count', 0)}",
            f"  Total chars  : {s.get('total_chars', 0):,}",
            f"  Alpha ratio  : {s.get('alpha_ratio', 0):.2f}",
            f"  Vowel ratio  : {s.get('vowel_ratio', 0):.2f}",
            f"  Seq. gaps    : {s.get('article_gaps', 0)}",
            f"  Broken paras : {s.get('broken_paragraphs', 0)}",
            f"  Duplicate sec: {s.get('duplicate_sections', 0)}",
        ]
        lines.append(sep)
        for issue in self.issues:
            lines.append(str(issue))
        if self.issues:
            lines.append(sep)
        return "\n".join(lines)


@dataclass
class FolderReport:
    """Batch quality result for a directory of text files."""
    reports:       list[QualityReport]
    folder:        Path
    total_files:   int
    valid_count:   int
    invalid_count: int

    @property
    def invalid_files(self) -> list[str]:
        return [r.file_name for r in self.reports if not r.valid]

    def summary(self) -> str:
        sep = "=" * 64
        lines = [
            sep,
            "  FOLDER QUALITY REPORT",
            f"  Folder  : {self.folder}",
            f"  Files   : {self.total_files}",
            f"  Valid   : {self.valid_count}",
            f"  Invalid : {self.invalid_count}",
            sep,
        ]
        for r in sorted(self.reports, key=lambda x: x.score):
            verdict = "VALID  " if r.valid else "INVALID"
            lines.append(
                f"  [{verdict}]  score={r.score:>3}  {r.file_name}"
            )
            for e in r.errors[:3]:
                lines.append(f"              {e.code}: {e.message}")
        lines.append(sep)
        return "\n".join(lines)


# -- OCR / character analysis -------------------------------------------------

# Albanian + common Latin vowels (for vowel-density check)
_VOWELS = frozenset("aeiouëAEIOUË")

# OCR ligatures that should have been cleaned
_LINGERING_LIGATURES = re.compile(r"[ﬀﬁﬂﬃﬄ]")

# Unicode replacement character (undecodable bytes)
_REPLACEMENT_CHAR = "�"

# Suspicious long token: > 30 chars, no hyphen or space (OCR word-merge)
_LONG_TOKEN_RE = re.compile(r"\b\S{31,}\b")

# Consonant run: 7+ consecutive consonants (no vowels) -- signs of garbling
_CONSONANT_RUN_RE = re.compile(
    r"[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{7,}"
)


def _ocr_stats(text: str) -> dict[str, Any]:
    """
    Measure character-level OCR quality indicators.

    Returns
    -------
    dict with keys:
        total_chars        : int
        total_words        : int
        alpha_ratio        : float  -- (letters + digits) / non-whitespace chars
        vowel_ratio        : float  -- vowels / alphabetic chars
                                       Albanian + English: ~0.38-0.45 expected
                                       < 0.15 indicates consonant-heavy garbling
        replacement_chars  : int    -- U+FFFD count
        lingering_ligatures: int
        long_token_count   : int    -- tokens > 30 chars
        consonant_runs     : int    -- 7+ consecutive consonants
    """
    non_ws  = [c for c in text if not c.isspace()]
    alpha   = sum(1 for c in non_ws if c.isalpha() or c.isdigit())
    letters = sum(1 for c in non_ws if c.isalpha())
    vowels  = sum(1 for c in non_ws if c in _VOWELS)
    total   = len(non_ws)

    return {
        "total_chars":         len(text),
        "total_words":         len(text.split()),
        "alpha_ratio":         alpha / total   if total   else 1.0,
        "vowel_ratio":         vowels / letters if letters else 1.0,
        "replacement_chars":   text.count(_REPLACEMENT_CHAR),
        "lingering_ligatures": len(_LINGERING_LIGATURES.findall(text)),
        "long_token_count":    len(_LONG_TOKEN_RE.findall(text)),
        "consonant_runs":      len(_CONSONANT_RUN_RE.findall(text)),
    }


# -- Article-sequence analysis ------------------------------------------------

def _article_gaps(article_matches: list[ArticleMatch]) -> list[str]:
    """
    Return a list of gap descriptions for non-consecutive article numbers.

    Only checks purely numeric article numbers (skips "5a", "5/a" variants).
    """
    nums: list[int] = []
    for art in article_matches:
        try:
            nums.append(int(art.number))
        except ValueError:
            pass   # letter-suffixed articles are exempt

    gaps: list[str] = []
    for i in range(len(nums) - 1):
        if nums[i + 1] - nums[i] > 1:
            gaps.append(f"gap between {nums[i]} and {nums[i + 1]}")
    return gaps


# -- Paragraph integrity analysis ---------------------------------------------

# Words that in Albanian/English typically END a sentence (not mid-sentence)
_SENTENCE_END_RE = re.compile(r"[.!?;]\s*$")

# Trailing hyphen: word broken at line end (truncation)
_TRAILING_HYPHEN_RE = re.compile(r"\w-\s*$")

# Paragraph that consists only of punctuation / whitespace
_NO_ALPHA_RE = re.compile(r"^[^a-zA-ZÀ-ɏ\d]+$")


def _broken_paragraphs(text: str) -> list[str]:
    """
    Return descriptions of potentially truncated or empty paragraphs.
    """
    broken: list[str] = []
    blocks = extract_structure(text)

    for art_block in blocks:
        for para in art_block.paragraphs:
            t = para.text.strip()

            if len(t) < 30:
                broken.append(
                    f"{art_block.match.label} / {para.label}: "
                    f"very short ({len(t)} chars)"
                )
                continue

            if _TRAILING_HYPHEN_RE.search(t):
                broken.append(
                    f"{art_block.match.label} / {para.label}: "
                    f"ends with hyphen (possible truncation)"
                )

            if _NO_ALPHA_RE.match(t):
                broken.append(
                    f"{art_block.match.label} / {para.label}: "
                    f"no alphabetic content"
                )

    return broken


# -- Duplicate section detection ----------------------------------------------

def _duplicate_sections(text: str) -> list[str]:
    """
    Detect article/paragraph bodies that appear more than once.

    Uses MD5 fingerprints of normalised paragraph text to avoid false
    positives from minor whitespace differences.
    """
    seen:  dict[str, str] = {}   # fingerprint -> first label
    dupes: list[str]      = []

    for art_block in extract_structure(text):
        for para in art_block.paragraphs:
            normalised = " ".join(para.text.lower().split())
            if len(normalised) < 40:
                continue  # skip trivially short paragraphs
            fp = md5(normalised.encode()).hexdigest()
            label = f"{art_block.match.label} / {para.label}"
            if fp in seen:
                dupes.append(f"{label} duplicates {seen[fp]}")
            else:
                seen[fp] = label

    return dupes


# -- Score computation --------------------------------------------------------

def _compute_score(
    issues: list[Issue],
    base:   int = 100,
) -> int:
    """Sum all penalties and return clamped score."""
    total = sum(i.penalty for i in issues)
    return max(0, min(100, base - total))


# -- Main check logic ---------------------------------------------------------

def check_text(
    text:      str,
    file_name: str = "",
) -> QualityReport:
    """
    Validate quality of extracted legal text.

    Parameters
    ----------
    text      : raw extracted text (output of pdf_loader)
    file_name : label for the report (usually the source file name)

    Returns
    -------
    QualityReport
    """
    issues:   list[Issue] = []
    errors:   list[Issue] = []
    warnings: list[Issue] = []

    # ── Compute raw measurements ──────────────────────────────────────────────
    ocr      = _ocr_stats(text)
    articles = find_articles(text)
    paras    = find_paragraphs(text)
    gaps     = _article_gaps(articles)
    broken   = _broken_paragraphs(text)
    dupes    = _duplicate_sections(text)

    stats: dict[str, Any] = {
        "total_chars":         ocr["total_chars"],
        "total_words":         ocr["total_words"],
        "alpha_ratio":         ocr["alpha_ratio"],
        "vowel_ratio":         ocr["vowel_ratio"],
        "replacement_chars":   ocr["replacement_chars"],
        "lingering_ligatures": ocr["lingering_ligatures"],
        "long_token_count":    ocr["long_token_count"],
        "consonant_runs":      ocr["consonant_runs"],
        "article_count":       len(articles),
        "paragraph_count":     len(paras),
        "article_gaps":        len(gaps),
        "broken_paragraphs":   len(broken),
        "duplicate_sections":  len(dupes),
    }

    def _err(code: str, msg: str, detail: str = "", penalty: int = 0) -> None:
        issues.append(Issue("error", code, msg, detail, penalty))

    def _warn(code: str, msg: str, detail: str = "", penalty: int = 0) -> None:
        issues.append(Issue("warning", code, msg, detail, penalty))

    # ── QC-1: Article detection ───────────────────────────────────────────────
    if not articles:
        _err("QC1_NO_ARTICLES",
             "No article headers found",
             "Document may be a non-law file, a scanned image, or severely garbled.",
             penalty=50)
    else:
        for gap in gaps[:5]:                 # cap at 5 to bound penalty
            _warn("QC1_ARTICLE_GAP",
                  "Article sequence gap",
                  gap,
                  penalty=5)

    # ── QC-2: Paragraph integrity ─────────────────────────────────────────────
    if articles and not paras:
        _err("QC2_NO_PARAGRAPHS",
             "No paragraph markers found despite articles being present",
             "Paragraph splitting will fall back to whole-body chunks.",
             penalty=15)

    for desc in broken[:5]:                  # cap at 5
        _warn("QC2_BROKEN_PARAGRAPH",
              "Potentially broken paragraph",
              desc,
              penalty=3)

    # ── QC-3: OCR quality ─────────────────────────────────────────────────────
    ar = ocr["alpha_ratio"]
    if ar < 0.40:
        _err("QC3_GARBLED_TEXT",
             f"Alpha ratio {ar:.2f} -- text is mostly non-alphabetic",
             "Likely a scanned page without OCR, or severe extraction failure.",
             penalty=35)
    elif ar < 0.55:
        _err("QC3_HIGH_OCR_ERROR",
             f"Alpha ratio {ar:.2f} -- high proportion of non-alphabetic characters",
             "Check for image-only pages or encoding corruption.",
             penalty=20)
    elif ar < 0.70:
        _warn("QC3_MEDIUM_OCR_ERROR",
              f"Alpha ratio {ar:.2f} -- moderately elevated non-alphabetic characters",
              penalty=8)

    vr = ocr["vowel_ratio"]
    if vr < 0.12:
        _err("QC3_LOW_VOWEL_DENSITY",
             f"Vowel density {vr:.2f} -- text appears consonant-heavy or garbled",
             "May indicate OCR producing random consonant sequences, or wrong character set.",
             penalty=25)
    elif vr < 0.22:
        _warn("QC3_LOW_VOWEL_DENSITY",
              f"Vowel density {vr:.2f} -- below expected range for Albanian/English (0.35-0.45)",
              penalty=10)

    if ocr["replacement_chars"] > 0:
        _err("QC3_REPLACEMENT_CHARS",
             f"{ocr['replacement_chars']} Unicode replacement character(s) (U+FFFD)",
             "Source bytes could not be decoded; check the PDF encoding.",
             penalty=5)

    if ocr["lingering_ligatures"] > 0:
        _warn("QC3_LIGATURES",
              f"{ocr['lingering_ligatures']} OCR ligature(s) not cleaned (ﬁ/ﬀ/ﬂ)",
              "Run pdf_loader or normalize.py to clean ligature artifacts.",
              penalty=3)

    for _ in range(min(ocr["long_token_count"], 5)):
        _warn("QC3_LONG_TOKEN",
              "Token longer than 30 characters (possible OCR word-merge)",
              penalty=2)

    for _ in range(min(ocr["consonant_runs"], 3)):
        _warn("QC3_CONSONANT_RUN",
              "7+ consecutive consonants detected (possible garbled text)",
              penalty=2)

    if ocr["total_chars"] < 200:
        _err("QC3_TOO_SHORT",
             f"Document has only {ocr['total_chars']} characters",
             "May be an empty extraction, a cover page, or an image-only PDF.",
             penalty=40)

    # ── QC-4: Duplicate sections ──────────────────────────────────────────────
    for desc in dupes[:4]:                   # cap at 4
        _warn("QC4_DUPLICATE",
              "Duplicate paragraph content detected",
              desc,
              penalty=3)

    # ── Partition and score ───────────────────────────────────────────────────
    errors   = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    score    = _compute_score(issues)
    valid    = score >= INVALID_THRESHOLD

    return QualityReport(
        file_name=file_name or "<raw text>",
        score=score,
        valid=valid,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


def check_file(path: str | Path) -> QualityReport:
    """
    Validate quality of a single .txt file.

    Parameters
    ----------
    path : path to a .txt file (output of pdf_loader.py)

    Returns
    -------
    QualityReport

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    text = p.read_text(encoding="utf-8", errors="replace")
    return check_text(text, file_name=p.name)


def check_folder(
    folder:  str | Path = "pdf_text",
    pattern: str        = "*.txt",
) -> FolderReport:
    """
    Validate quality of all .txt files in a directory.

    Parameters
    ----------
    folder  : directory containing .txt files from pdf_loader.py
    pattern : glob pattern for input files (default "*.txt")

    Returns
    -------
    FolderReport

    Raises
    ------
    FileNotFoundError
        If the folder does not exist.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = sorted(folder.glob(pattern))
    reports: list[QualityReport] = []

    for f in files:
        try:
            reports.append(check_file(f))
        except Exception as exc:
            # Unreadable file → score 0
            reports.append(QualityReport(
                file_name=f.name,
                score=0,
                valid=False,
                errors=[Issue("error", "QC_UNREADABLE",
                              f"Could not read file: {exc}", penalty=100)],
                warnings=[],
                stats={},
            ))

    valid_count   = sum(1 for r in reports if r.valid)
    invalid_count = len(reports) - valid_count

    return FolderReport(
        reports=reports,
        folder=folder,
        total_files=len(reports),
        valid_count=valid_count,
        invalid_count=invalid_count,
    )


# -- CLI ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import logging
    import sys

    logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)

    parser = argparse.ArgumentParser(
        prog="python -m rag.pdf_quality_check",
        description="Validate extracted legal text quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
score >= {INVALID_THRESHOLD}: VALID   (safe to ingest)
score <  {INVALID_THRESHOLD}: INVALID (needs review or re-extraction)

examples:
  # Check entire pdf_text/ folder
  python -m rag.pdf_quality_check

  # Check specific folder
  python -m rag.pdf_quality_check --folder docs/txt

  # Check a single file
  python -m rag.pdf_quality_check --file law.txt

  # Show only invalid files
  python -m rag.pdf_quality_check --invalid-only

  # Suppress warnings in output
  python -m rag.pdf_quality_check --errors-only
        """,
    )
    parser.add_argument("--folder",       default="pdf_text",   metavar="DIR",
                        help="Folder to check (default: pdf_text)")
    parser.add_argument("--file",  "-f",  default=None,         metavar="FILE",
                        help="Check a single .txt file")
    parser.add_argument("--invalid-only", action="store_true",
                        help="Print only INVALID file reports")
    parser.add_argument("--errors-only",  action="store_true",
                        help="Print only errors, suppress warnings")
    args = parser.parse_args()

    if args.file:
        try:
            report = check_file(args.file)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(2)
        print(report.summary())
        sys.exit(0 if report.valid else 1)

    # Folder mode
    try:
        folder_report = check_folder(args.folder)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    for r in folder_report.reports:
        if args.invalid_only and r.valid:
            continue
        lines = [r.summary()]
        if args.errors_only:
            # Rebuild summary without warnings
            r2 = QualityReport(
                file_name=r.file_name, score=r.score, valid=r.valid,
                errors=r.errors, warnings=[], stats=r.stats,
            )
            lines = [r2.summary()]
        print("\n".join(lines))
        print()

    print(folder_report.summary())
    sys.exit(0 if folder_report.invalid_count == 0 else 1)
