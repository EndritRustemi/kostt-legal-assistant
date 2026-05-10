"""
build_pipeline.py -- Full Legal PDF Ingestion Pipeline

Orchestrates the four-stage pipeline that turns raw PDF files into a
validated, deduplicated chunk corpus ready for embedding and retrieval.

Stages
------
  [1] PDF Extraction   pdf_loader  -- pdfs_raw/*.pdf  -> pdf_text/*.txt
  [2] Segmentation     segment_law -- pdf_text/*.txt  -> chunks in memory
  [3] Deduplication                -- drop repeated chunk IDs (keep first)
  [4] Validation                   -- article + paragraph + text required

Output
------
  processed_chunks.json  -- merged, deduplicated, validated corpus

Fail-safety
-----------
  - A single PDF failing extraction does not abort the run; other files
    continue and the failure is recorded in the result.
  - A single TXT failing segmentation is skipped; processing continues.
  - Validation errors are logged in full but do NOT prevent the output
    file from being written -- the caller sees ok=False and can decide.
  - The only hard abort is when the output path is not writable, or when
    zero chunks survive all stages (nothing to write).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Pipeline components (relative to the legal-rag/ directory)
from rag.pdf_loader  import extract_folder, ExtractionResult
from rag.segment_law import segment_file

_LOG_FMT  = "%(asctime)s | %(levelname)-8s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


# -- Result types -------------------------------------------------------------

@dataclass
class FileReport:
    """Per-file extraction + segmentation stats."""
    name:       str
    txt_path:   Path | None
    chunks:     int
    ok:         bool
    warnings:   list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "OK" if self.ok else "FAIL"
        return (
            f"  [{status}] {self.name:<40}  {self.chunks:>5} chunk(s)"
            + (f"  WARN: {self.warnings[0]}" if self.warnings else "")
        )


@dataclass
class ValidationError:
    index:    int
    chunk_id: str
    field:    str
    message:  str

    def __str__(self) -> str:
        return f"  chunk #{self.index} (id={self.chunk_id!r}): [{self.field}] {self.message}"


@dataclass
class PipelineResult:
    """Complete pipeline execution summary."""
    # Per-file breakdown
    file_reports:   list[FileReport]

    # Counts
    pdfs_found:     int
    pdfs_extracted: int
    txts_found:     int
    txts_segmented: int
    chunks_raw:     int       # before deduplication
    chunks_deduped: int       # duplicates removed
    chunks_final:   int       # written to output

    # Validation
    validation_errors: list[ValidationError]

    # Output
    output_path: Path
    elapsed_s:   float
    ok:          bool         # True iff no extraction failures AND no validation errors

    def summary(self) -> str:
        sep = "-" * 64
        lines = [
            sep,
            "  PIPELINE SUMMARY",
            sep,
            f"  PDFs found         : {self.pdfs_found}",
            f"  PDFs extracted     : {self.pdfs_extracted}",
            f"  TXT files found    : {self.txts_found}",
            f"  TXT files segmented: {self.txts_segmented}",
            f"  Chunks (raw)       : {self.chunks_raw}",
            f"  Duplicates removed : {self.chunks_deduped}",
            f"  Chunks (final)     : {self.chunks_final}",
            f"  Validation errors  : {len(self.validation_errors)}",
            f"  Output             : {self.output_path}",
            f"  Elapsed            : {self.elapsed_s:.1f}s",
            sep,
            "  Result : " + ("PASS" if self.ok else "FAIL"),
            sep,
        ]
        if self.validation_errors:
            lines.insert(-2, "  Validation errors:")
            for e in self.validation_errors[:20]:
                lines.insert(-2, str(e))
            if len(self.validation_errors) > 20:
                lines.insert(-2,
                    f"  ... and {len(self.validation_errors) - 20} more")
        return "\n".join(lines)


# -- Inline validation --------------------------------------------------------
# validate_chunks.validate() checks source/semantic that segment_law does not
# produce, so we validate only the fields this pipeline guarantees.

_REQUIRED_STR_FIELDS = ("id", "text", "article", "paragraph", "document_title")


def _validate(chunks: list[dict[str, Any]]) -> list[ValidationError]:
    """
    Check every chunk for the fields this pipeline guarantees.

    Rules
    -----
      V1  id            -- present, non-empty string
      V2  text          -- present, non-whitespace
      V3  article       -- present, non-empty
      V4  paragraph     -- present, non-empty
      V5  document_title-- present, non-empty
      V6  no duplicate IDs
    """
    errors:  list[ValidationError] = []
    seen:    dict[str, int]        = {}   # id -> first-seen index
    dup_ids: set[str]              = set()

    for i, chunk in enumerate(chunks):
        cid = chunk.get("id", "") if isinstance(chunk, dict) else ""
        if not isinstance(cid, str):
            cid = str(cid)

        if not isinstance(chunk, dict):
            errors.append(ValidationError(i, "<not-a-dict>", "type",
                                          f"expected dict, got {type(chunk).__name__}"))
            continue

        # V1-V5: required string fields
        for f in _REQUIRED_STR_FIELDS:
            val = chunk.get(f)
            if not isinstance(val, str) or not val.strip():
                errors.append(ValidationError(
                    i, cid or "<missing>", f,
                    f"'{f}' missing or empty",
                ))

        # V6: duplicate IDs
        if cid:
            if cid in seen:
                if cid not in dup_ids:
                    errors.append(ValidationError(
                        i, cid, "id",
                        f"duplicate (first seen at index {seen[cid]})",
                    ))
                    dup_ids.add(cid)
            else:
                seen[cid] = i

    return errors


# -- Deduplication ------------------------------------------------------------

def _deduplicate(
    chunks: list[dict[str, Any]],
    log:    logging.Logger,
) -> tuple[list[dict[str, Any]], int]:
    """
    Remove chunks with repeated IDs, keeping the first occurrence.

    Returns (deduplicated_list, count_removed).
    """
    seen:     dict[str, int]         = {}
    result:   list[dict[str, Any]]   = []
    removed:  list[str]              = []

    for chunk in chunks:
        cid = chunk.get("id", "")
        if not isinstance(cid, str) or not cid:
            result.append(chunk)   # keep; validation will flag missing ID
            continue
        if cid in seen:
            removed.append(cid)
        else:
            seen[cid] = 1
            result.append(chunk)

    if removed:
        log.warning(
            "Deduplication removed %d chunk(s). Repeated IDs: %s",
            len(removed),
            removed[:10],
        )
    return result, len(removed)


# -- Logging setup ------------------------------------------------------------

def _setup_logging(log_file: Path | None = None) -> logging.Logger:
    """
    Configure the root logger with console + optional file handler.
    Returns the pipeline-specific logger.
    """
    root = logging.getLogger()
    if root.handlers:
        root.handlers.clear()

    fmt = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    root.setLevel(logging.INFO)
    return logging.getLogger("legal.pipeline")


# -- Main pipeline ------------------------------------------------------------

def run_pipeline(
    pdfs_dir:    str | Path = "pdfs_raw",
    txt_dir:     str | Path = "pdf_text",
    output:      str | Path = "processed_chunks.json",
    encoding:    str        = "utf-8",
    overwrite:   bool       = False,
    log_file:    str | Path | None = None,
) -> PipelineResult:
    """
    Run the full four-stage ingestion pipeline.

    Parameters
    ----------
    pdfs_dir  : folder containing source PDFs (skipped if absent)
    txt_dir   : intermediate .txt output from pdf_loader (created if absent)
    output    : path for the final processed_chunks.json
    encoding  : file encoding throughout (default utf-8)
    overwrite : re-extract and re-segment files that already exist
    log_file  : optional file path for persistent log output

    Returns
    -------
    PipelineResult
        Full execution summary. Check .ok and .summary() for status.
    """
    t_start = time.monotonic()
    log = _setup_logging(Path(log_file) if log_file else None)

    pdfs_dir = Path(pdfs_dir)
    txt_dir  = Path(txt_dir)
    output   = Path(output)

    file_reports:   list[FileReport]   = []
    all_chunks:     list[dict]         = []
    pdfs_found      = 0
    pdfs_extracted  = 0
    txts_found      = 0
    txts_segmented  = 0

    # ── Stage 1: PDF Extraction ───────────────────────────────────────────────
    sep = "-" * 64
    log.info(sep)
    log.info("[STAGE 1/4]  PDF Extraction  (%s -> %s)", pdfs_dir, txt_dir)
    log.info(sep)

    if not pdfs_dir.exists():
        log.warning(
            "pdfs_dir '%s' does not exist -- skipping extraction, "
            "will segment any existing TXT files in '%s'",
            pdfs_dir, txt_dir,
        )
    else:
        pdf_files = sorted(pdfs_dir.glob("*.pdf"))
        pdfs_found = len(pdf_files)
        log.info("Found %d PDF(s) in %s", pdfs_found, pdfs_dir)

        if pdfs_found:
            extraction_results: list[ExtractionResult] = extract_folder(
                pdfs_dir, txt_dir,
                overwrite=overwrite,
                encoding=encoding,
            )
            for r in extraction_results:
                ok = r.ok
                if ok:
                    pdfs_extracted += 1
                skipped = any("skipped" in w for w in r.warnings)
                log.info(
                    "  %-40s  %6d chars  %d pages  %d noise block(s)%s",
                    r.pdf_path.name,
                    r.char_count,
                    r.page_count,
                    r.noise_blocks_removed,
                    "  [skipped]" if skipped else ("  [FAIL]" if not ok else ""),
                )
                for w in r.warnings:
                    if "skipped" not in w:
                        log.warning("    %s: %s", r.pdf_path.name, w)

            log.info(
                "Stage 1 result: %d extracted, %d skipped, %d failed",
                pdfs_extracted,
                sum(1 for r in extraction_results
                    if any("skipped" in w for w in r.warnings)),
                sum(1 for r in extraction_results if not r.ok),
            )

    # ── Stage 2: Segmentation ─────────────────────────────────────────────────
    log.info(sep)
    log.info("[STAGE 2/4]  Segmentation  (%s -> memory)", txt_dir)
    log.info(sep)

    txt_files = sorted(txt_dir.glob("*.txt")) if txt_dir.exists() else []
    txts_found = len(txt_files)

    if not txt_files:
        log.error(
            "No .txt files found in '%s'. "
            "Ensure pdf_loader ran successfully or place .txt files there manually.",
            txt_dir,
        )
        return PipelineResult(
            file_reports=file_reports,
            pdfs_found=pdfs_found, pdfs_extracted=pdfs_extracted,
            txts_found=0, txts_segmented=0,
            chunks_raw=0, chunks_deduped=0, chunks_final=0,
            validation_errors=[],
            output_path=output, elapsed_s=time.monotonic() - t_start,
            ok=False,
        )

    log.info("Found %d TXT file(s) in %s", txts_found, txt_dir)

    for txt in txt_files:
        try:
            chunks = segment_file(txt, output_path=None, encoding=encoding)
        except Exception as exc:
            log.error("  FAIL  %s: %s", txt.name, exc)
            file_reports.append(FileReport(
                name=txt.name, txt_path=txt, chunks=0, ok=False,
                warnings=[str(exc)],
            ))
            continue

        n = len(chunks)
        txts_segmented += 1
        all_chunks.extend(chunks)

        log.info("  %-44s  %5d chunk(s)", txt.name, n)
        file_reports.append(FileReport(
            name=txt.name, txt_path=txt, chunks=n, ok=True,
        ))

    chunks_raw = len(all_chunks)
    log.info(
        "Stage 2 result: %d segmented, %d failed, %d total chunk(s)",
        txts_segmented,
        txts_found - txts_segmented,
        chunks_raw,
    )

    if not all_chunks:
        log.error(
            "Zero chunks produced. "
            "Check that the TXT files contain 'Neni N' or 'Article N' headings."
        )
        return PipelineResult(
            file_reports=file_reports,
            pdfs_found=pdfs_found, pdfs_extracted=pdfs_extracted,
            txts_found=txts_found, txts_segmented=txts_segmented,
            chunks_raw=0, chunks_deduped=0, chunks_final=0,
            validation_errors=[],
            output_path=output, elapsed_s=time.monotonic() - t_start,
            ok=False,
        )

    # ── Stage 3: Deduplication ────────────────────────────────────────────────
    log.info(sep)
    log.info("[STAGE 3/4]  Deduplication")
    log.info(sep)
    log.info("  Chunks before : %d", chunks_raw)

    deduped_chunks, n_removed = _deduplicate(all_chunks, log)
    chunks_deduped = n_removed
    chunks_final   = len(deduped_chunks)

    log.info("  Duplicates    : %d", n_removed)
    log.info("  Chunks after  : %d", chunks_final)

    # ── Stage 4: Validation ───────────────────────────────────────────────────
    log.info(sep)
    log.info("[STAGE 4/4]  Validation  (%d chunks)", chunks_final)
    log.info(sep)

    val_errors = _validate(deduped_chunks)

    # Report per-rule counts
    by_field: dict[str, int] = {}
    for e in val_errors:
        by_field[e.field] = by_field.get(e.field, 0) + 1

    checks = [
        ("id",             "ID present and non-empty"),
        ("text",           "Text non-whitespace"),
        ("article",        "Article reference present"),
        ("paragraph",      "Paragraph reference present"),
        ("document_title", "Document title present"),
    ]
    for field_name, label in checks:
        count = by_field.get(field_name, 0)
        flag  = "FAIL" if count else "PASS"
        line  = f"  [{flag}]  {label}"
        if count:
            line += f"  ({count} violation(s))"
        log.info(line)

    dup_val = by_field.get("id", 0)
    log.info("  [%s]  No duplicate IDs", "FAIL" if dup_val else "PASS")

    if val_errors:
        log.warning("Validation found %d error(s):", len(val_errors))
        for e in val_errors[:20]:
            log.warning("  %s", e)
        if len(val_errors) > 20:
            log.warning("  ... and %d more", len(val_errors) - 20)
    else:
        log.info("  Validation PASSED -- all %d chunks are well-formed", chunks_final)

    # ── Write output ──────────────────────────────────────────────────────────
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(deduped_chunks, ensure_ascii=False, indent=2),
        encoding=encoding,
    )

    elapsed = time.monotonic() - t_start
    ok = (
        txts_segmented > 0
        and chunks_final > 0
        and len(val_errors) == 0
    )

    log.info(sep)
    log.info(
        "Output: %s  (%d chunk(s), %.1fs elapsed)",
        output, chunks_final, elapsed,
    )
    log.info("Result: %s", "PASS" if ok else "FAIL")
    log.info(sep)

    return PipelineResult(
        file_reports=file_reports,
        pdfs_found=pdfs_found, pdfs_extracted=pdfs_extracted,
        txts_found=txts_found, txts_segmented=txts_segmented,
        chunks_raw=chunks_raw, chunks_deduped=chunks_deduped,
        chunks_final=chunks_final,
        validation_errors=val_errors,
        output_path=output,
        elapsed_s=elapsed,
        ok=ok,
    )


# -- CLI ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="python build_pipeline.py",
        description="Full legal PDF ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
stages:
  1  pdf_loader  -- extract text from PDFs in pdfs_raw/
  2  segment_law -- split TXT files into article/paragraph chunks
  3  dedup       -- remove chunks with repeated IDs (keep first)
  4  validate    -- verify article + paragraph + text on every chunk

examples:
  python build_pipeline.py
  python build_pipeline.py --pdfs docs/laws --output data/corpus.json
  python build_pipeline.py --overwrite --log pipeline.log
  python build_pipeline.py --skip-extraction  # segment existing TXTs only
        """,
    )
    parser.add_argument("--pdfs",   default="pdfs_raw", metavar="DIR",
                        help="Source PDF folder (default: pdfs_raw)")
    parser.add_argument("--txt",    default="pdf_text", metavar="DIR",
                        help="Intermediate TXT folder (default: pdf_text)")
    parser.add_argument("--output", default="processed_chunks.json", metavar="FILE",
                        help="Output JSON path (default: processed_chunks.json)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract and re-segment already-processed files")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip Stage 1 (PDF extraction); segment existing TXTs only")
    parser.add_argument("--log", default=None, metavar="FILE",
                        help="Write log to FILE in addition to stderr")
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    pdfs_dir = "__skip__" if args.skip_extraction else args.pdfs

    result = run_pipeline(
        pdfs_dir  = pdfs_dir,
        txt_dir   = args.txt,
        output    = args.output,
        overwrite = args.overwrite,
        log_file  = args.log,
    )

    # Print final summary to stdout
    print()
    print(result.summary())
    sys.exit(0 if result.ok else 1)
