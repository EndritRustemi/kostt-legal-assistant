"""
validate_chunks.py -- Chunk Dataset Integrity Validator

Validates processed_chunks.json before it is loaded into the retrieval
pipeline. Fails fast on critical corruption; collects all errors before
reporting so the full picture is visible in one run.

Checks performed
----------------
  C1  id present and non-empty string
  C2  text present, non-empty, and non-whitespace-only
  C3  legal_reference present with non-empty article and paragraph
  C4  article and paragraph top-level fields present and non-empty
  C5  article / paragraph consistency: top-level matches legal_reference
  C6  no duplicate ids across the dataset
  C7  source present (dict with document_title, or non-empty string)
  C8  semantic present with keywords (list) and summary (str)

Exit codes
----------
  0  all checks pass
  1  one or more errors found
  2  file not found / unreadable / not valid JSON
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# -- Error model --------------------------------------------------------------

@dataclass
class ChunkError:
    index:    int    # 0-based position in the JSON array
    chunk_id: str    # value of chunk["id"] if readable, else "<missing>"
    code:     str    # e.g. "C2", "C6"
    message:  str    # human-readable description

    def __str__(self) -> str:
        return f"[{self.code}] chunk #{self.index} (id={self.chunk_id!r}): {self.message}"


@dataclass
class ValidationResult:
    errors:         list[ChunkError]
    total_chunks:   int
    duplicates:     list[str]               # ids that appear more than once
    missing_fields: dict[str, int] = field(default_factory=dict)  # field -> count
    passed:         bool = True

    @property
    def error_count(self) -> int:
        return len(self.errors)

    def summary(self) -> str:
        sep = "-" * 60
        lines = [
            sep,
            f"  Total chunks    : {self.total_chunks}",
            f"  Errors found    : {self.error_count}",
            f"  Duplicate IDs   : {len(self.duplicates)}",
        ]
        if self.missing_fields:
            lines.append("  Missing fields  :")
            for fname, cnt in sorted(self.missing_fields.items()):
                lines.append(f"      {fname:<26}: {cnt} chunk(s)")
        if self.duplicates:
            lines.append("  Duplicate IDs   :")
            for dup in self.duplicates[:10]:
                lines.append(f"      {dup!r}")
            if len(self.duplicates) > 10:
                lines.append(f"      ... and {len(self.duplicates) - 10} more")
        lines.append(sep)
        lines.append("  Result : " + ("PASS" if self.passed else "FAIL"))
        lines.append(sep)
        return "\n".join(lines)


# -- Per-chunk checks ---------------------------------------------------------

def _chunk_id(chunk: Any) -> str:
    """Best-effort extraction of chunk id for error messages."""
    if isinstance(chunk, dict):
        v = chunk.get("id")
        if isinstance(v, str) and v.strip():
            return v
    return "<missing>"


def _check_chunk(
    chunk:  Any,
    index:  int,
    errors: list[ChunkError],
    mf:     dict[str, int],
) -> str:
    """
    Run all per-chunk checks (C1-C8).
    Returns the chunk id (or '<missing>') for use in the duplicate check.
    """
    if not isinstance(chunk, dict):
        errors.append(ChunkError(
            index, "<not-a-dict>", "C0",
            f"expected dict, got {type(chunk).__name__}",
        ))
        return "<not-a-dict>"

    cid = _chunk_id(chunk)

    # C1 -- id
    raw_id = chunk.get("id")
    if not isinstance(raw_id, str) or not raw_id.strip():
        errors.append(ChunkError(index, cid, "C1", "missing or empty 'id'"))
        mf["id"] = mf.get("id", 0) + 1

    # C2 -- text
    raw_text = chunk.get("text")
    if not isinstance(raw_text, str) or not raw_text.strip():
        errors.append(ChunkError(index, cid, "C2", "missing or whitespace-only 'text'"))
        mf["text"] = mf.get("text", 0) + 1

    # C3 -- legal_reference
    lr = chunk.get("legal_reference")
    if not isinstance(lr, dict):
        errors.append(ChunkError(index, cid, "C3",
                                 "'legal_reference' missing or not a dict"))
        mf["legal_reference"] = mf.get("legal_reference", 0) + 1
    else:
        if not isinstance(lr.get("article"), str) or not lr["article"].strip():
            errors.append(ChunkError(index, cid, "C3",
                                     "'legal_reference.article' missing or empty"))
            mf["legal_reference.article"] = mf.get("legal_reference.article", 0) + 1
        if not isinstance(lr.get("paragraph"), str) or not lr["paragraph"].strip():
            errors.append(ChunkError(index, cid, "C3",
                                     "'legal_reference.paragraph' missing or empty"))
            mf["legal_reference.paragraph"] = mf.get("legal_reference.paragraph", 0) + 1

    # C4 -- top-level article / paragraph
    top_article   = chunk.get("article")
    top_paragraph = chunk.get("paragraph")

    if not isinstance(top_article, str) or not top_article.strip():
        errors.append(ChunkError(index, cid, "C4",
                                 "top-level 'article' missing or empty"))
        mf["article"] = mf.get("article", 0) + 1

    if not isinstance(top_paragraph, str) or not top_paragraph.strip():
        errors.append(ChunkError(index, cid, "C4",
                                 "top-level 'paragraph' missing or empty"))
        mf["paragraph"] = mf.get("paragraph", 0) + 1

    # C5 -- consistency: top-level must match legal_reference
    if (
        isinstance(lr, dict)
        and isinstance(top_article, str) and top_article.strip()
        and isinstance(top_paragraph, str) and top_paragraph.strip()
    ):
        if lr.get("article", "").strip() != top_article.strip():
            errors.append(ChunkError(
                index, cid, "C5",
                f"article mismatch: top={top_article!r} "
                f"vs legal_reference={lr.get('article')!r}",
            ))
        if lr.get("paragraph", "").strip() != top_paragraph.strip():
            errors.append(ChunkError(
                index, cid, "C5",
                f"paragraph mismatch: top={top_paragraph!r} "
                f"vs legal_reference={lr.get('paragraph')!r}",
            ))

    # C7 -- source
    src = chunk.get("source")
    if src is None:
        errors.append(ChunkError(index, cid, "C7", "'source' missing"))
        mf["source"] = mf.get("source", 0) + 1
    elif isinstance(src, dict):
        title = src.get("document_title")
        if not isinstance(title, str) or not title.strip():
            errors.append(ChunkError(index, cid, "C7",
                                     "'source.document_title' missing or empty"))
            mf["source.document_title"] = mf.get("source.document_title", 0) + 1
    elif not isinstance(src, str) or not src.strip():
        errors.append(ChunkError(index, cid, "C7",
                                 "'source' must be a non-empty string or dict"))
        mf["source"] = mf.get("source", 0) + 1

    # C8 -- semantic
    sem = chunk.get("semantic")
    if not isinstance(sem, dict):
        errors.append(ChunkError(index, cid, "C8",
                                 "'semantic' missing or not a dict"))
        mf["semantic"] = mf.get("semantic", 0) + 1
    else:
        if not isinstance(sem.get("keywords"), list) or not sem["keywords"]:
            errors.append(ChunkError(index, cid, "C8",
                                     "'semantic.keywords' missing or empty list"))
            mf["semantic.keywords"] = mf.get("semantic.keywords", 0) + 1
        if not isinstance(sem.get("summary"), str) or not sem["summary"].strip():
            errors.append(ChunkError(index, cid, "C8",
                                     "'semantic.summary' missing or empty"))
            mf["semantic.summary"] = mf.get("semantic.summary", 0) + 1

    return cid


# -- Public API ---------------------------------------------------------------

def validate(chunks: list[Any]) -> ValidationResult:
    """
    Validate a list of chunk dicts loaded from processed_chunks.json.

    Parameters
    ----------
    chunks : list
        Parsed JSON array -- each element is expected to be a dict.

    Returns
    -------
    ValidationResult
        .errors         -- all ChunkError objects (one per violation)
        .total_chunks   -- len(chunks)
        .duplicates     -- ids that appear more than once
        .missing_fields -- field-name -> count of chunks missing that field
        .passed         -- True iff error_count == 0
    """
    errors:  list[ChunkError] = []
    mf:      dict[str, int]   = {}
    seen:    dict[str, int]   = {}   # id -> first-seen index
    dup_ids: set[str]         = set()

    for i, chunk in enumerate(chunks):
        cid = _check_chunk(chunk, i, errors, mf)

        # C6 -- duplicate id
        if cid not in ("<missing>", "<not-a-dict>"):
            if cid in seen:
                if cid not in dup_ids:
                    errors.append(ChunkError(
                        i, cid, "C6",
                        f"duplicate id (first seen at index {seen[cid]})",
                    ))
                    dup_ids.add(cid)
            else:
                seen[cid] = i

    return ValidationResult(
        errors=errors,
        total_chunks=len(chunks),
        duplicates=sorted(dup_ids),
        missing_fields=mf,
        passed=len(errors) == 0,
    )


def validate_file(path: str | Path) -> ValidationResult:
    """
    Load and validate a processed_chunks.json file.

    Raises
    ------
    SystemExit(2)
        If the file does not exist, cannot be read, is not valid JSON,
        or is not a JSON array. (Fail-fast on corrupt dataset.)
    """
    p = Path(path)
    if not p.exists():
        print(f"ERROR: file not found: {p}", file=sys.stderr)
        sys.exit(2)

    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: cannot read {p}: {exc}", file=sys.stderr)
        sys.exit(2)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"ERROR: {p} is not valid JSON: {exc}", file=sys.stderr)
        sys.exit(2)

    if not isinstance(data, list):
        print(
            f"ERROR: {p} must contain a JSON array at the top level, "
            f"got {type(data).__name__}",
            file=sys.stderr,
        )
        sys.exit(2)

    return validate(data)


# -- CLI ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m rag.validate_chunks",
        description="Validate processed_chunks.json integrity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python -m rag.validate_chunks
  python -m rag.validate_chunks data/corpus.json
  python -m rag.validate_chunks data/corpus.json --errors-only
  python -m rag.validate_chunks data/corpus.json --max-errors 20
        """,
    )
    parser.add_argument(
        "file",
        nargs="?",
        default="processed_chunks.json",
        help="Path to chunks JSON file (default: processed_chunks.json)",
    )
    parser.add_argument(
        "--errors-only",
        action="store_true",
        help="Print only errors, suppress summary statistics",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=0,
        metavar="N",
        help="Stop printing errors after N (0 = unlimited)",
    )
    args = parser.parse_args()

    result = validate_file(args.file)

    if result.errors:
        limit = args.max_errors or len(result.errors)
        for err in result.errors[:limit]:
            print(err)
        if len(result.errors) > limit:
            print(f"... {len(result.errors) - limit} more error(s) not shown "
                  f"(use --max-errors 0 to see all)")

    if not args.errors_only:
        print()
        print(result.summary())

    sys.exit(0 if result.passed else 1)
