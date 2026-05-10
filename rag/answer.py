"""
answer.py -- Legal Answer Orchestrator

Orchestrates the full pipeline:
    question -> memory -> query_rewriter -> retrieve -> cite -> memory

Flow
----
1. Load last 3 turns from ConversationMemory
2. Rewrite the question into a standalone query (query_rewriter.py)
3. Retrieve chunks with the rewritten query (retrieve.py)
4. Generate a cited legal answer (cite.py)
5. Enforce output policy (policy.py)
6. Save the interaction to ConversationMemory

Guarantees
----------
- Never raises to the caller (all errors are captured in the result)
- Works with empty memory (no history -> no rewriting attempted)
- Rewrite failures are non-fatal: the original question is used as fallback
- Every run is logged (retrieved IDs, rewritten query, rejections, timing)
- Minimum chunk threshold enforced before the citation engine is called
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Literal

from rag.retrieve import LegalRetriever, DEFAULT_MODEL, DEFAULT_THRESHOLD
from rag.cite import generate_cited_answer, INSUFFICIENT_BASIS
from rag.policy import enforce, PolicyViolation, policy_error_result
from rag.query_rewriter import rewrite as _rewrite_query, Turn
from rag.debug import dprint

# ── Logging setup ──────────────────────────────────────────────────────────────

_LOG_FMT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
log = logging.getLogger("legal.answer")


def configure_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
) -> None:
    log.setLevel(level)
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT))
        log.addHandler(fh)
        log.info("File logging enabled -> %s", log_file)


# ── Conversation memory ────────────────────────────────────────────────────────

class ConversationMemory:
    """
    Stores conversation turns and supplies recent history for query rewriting.

    Parameters
    ----------
    path : str | Path | None
        JSON file path for persistent storage across sessions.
        Pass None for in-process (session-only) memory.
    max_context : int
        Number of recent turns passed to the query rewriter (default 3).
    """

    _MAX_STORED = 50   # total turns kept in the file; oldest are dropped

    def __init__(
        self,
        path: str | Path | None = None,
        max_context: int = 3,
    ) -> None:
        self._path: Path | None = Path(path) if path else None
        self._max_context = max_context
        self._turns: list[dict[str, str]] = []
        if self._path and self._path.exists():
            self._load()

    # -- Public interface ──────────────────────────────────────────────────────

    def recent(self) -> list[Turn]:
        """Return the last ``max_context`` turns as Turn objects."""
        return [
            Turn(t["question"], t["answer"])
            for t in self._turns[-self._max_context:]
        ]

    def add(self, question: str, answer: str) -> None:
        """Append a turn and persist if a file path was configured."""
        self._turns.append({"question": question, "answer": answer})
        if len(self._turns) > self._MAX_STORED:
            self._turns = self._turns[-self._MAX_STORED:]
        self._save()

    def clear(self) -> None:
        """Remove all stored turns and delete the backing file if present."""
        self._turns.clear()
        if self._path and self._path.exists():
            try:
                self._path.unlink()
            except OSError as exc:
                log.warning("Memory clear: could not delete file: %s", exc)

    def __len__(self) -> int:
        return len(self._turns)

    # -- Private helpers ───────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
            if isinstance(data, list):
                self._turns = [
                    t for t in data
                    if isinstance(t, dict)
                    and isinstance(t.get("question"), str)
                    and isinstance(t.get("answer"), str)
                ]
        except Exception as exc:
            log.warning("Memory load failed (%s) — starting empty.", exc)
            self._turns = []

    def _save(self) -> None:
        if not self._path:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._turns, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("Memory save failed: %s", exc)


# ── Result schema ──────────────────────────────────────────────────────────────

class AnswerResult(dict):
    """
    Typed dict-like result returned by run().

    Keys
    ----
    status             : "ok" | "insufficient_basis" | "retrieval_error"
                         | "citation_error" | "error"
    question           : original question string (as the user typed it)
    rewritten_question : standalone query sent to the retriever; equals
                         ``question`` when no rewriting was needed
    was_rewritten      : True if the query rewriter made a change
    chunks_retrieved   : number of chunks that passed the threshold
    chunk_ids          : list of retrieved chunk IDs (strings)
    answer             : full cite.py output dict (present when status == "ok")
    rejection_reason   : internal explanation when status != "ok" (for logging)
    user_message       : always a non-empty string safe to display to the user;
                         on success this is the legal conclusion; on any failure
                         it is a clear, specific error message
    elapsed_ms         : total pipeline time in milliseconds
    """


def _make_result(
    status: Literal["ok", "insufficient_basis", "retrieval_error",
                    "citation_error", "error"],
    question: str,
    rewritten: str,
    was_rewritten: bool,
    chunks: list[dict],
    answer: dict | None,
    reason: str | None,
    elapsed_ms: int,
    user_message: str = "",
) -> AnswerResult:
    return AnswerResult(
        status=status,
        question=question,
        rewritten_question=rewritten,
        was_rewritten=was_rewritten,
        chunks_retrieved=len(chunks),
        chunk_ids=[str(c.get("id", "unknown")) for c in chunks],
        answer=answer,
        rejection_reason=reason,
        user_message=user_message or reason or "An unexpected error occurred.",
        elapsed_ms=elapsed_ms,
    )


# ── Pipeline ───────────────────────────────────────────────────────────────────

MIN_CHUNKS = 2


def run(
    question: str,
    retriever: LegalRetriever,
    api_key: str,
    top_k: int = 5,
    threshold: float = DEFAULT_THRESHOLD,
    memory: ConversationMemory | None = None,
) -> AnswerResult:
    """
    Execute the full memory -> rewrite -> retrieval -> citation pipeline.

    Parameters
    ----------
    question  : user's legal question (may contain references to prior turns)
    retriever : a pre-built LegalRetriever instance (corpus already indexed)
    api_key   : Anthropic API key used by both the rewriter and citation engine
    top_k     : maximum chunks to retrieve (default 5)
    threshold : minimum retrieval score to accept a chunk
    memory    : ConversationMemory instance; pass None to skip memory entirely

    Returns
    -------
    AnswerResult
        Always returns a dict -- never raises.

    Notes
    -----
    - If the query rewriter fails, the original question is used and a
      WARNING is logged. The pipeline continues normally.
    - Memory save failures are logged but do not affect the returned result.
    """
    t0 = time.monotonic()

    # ── Guard: empty question ──────────────────────────────────────────────────
    if not question or not question.strip():
        reason = "Empty or blank question received."
        log.warning("REJECTED | reason='%s'", reason)
        return _make_result(
            "error", question, question, False, [], None, reason, _ms(t0),
            user_message="Please enter a legal question.",
        )

    q = question.strip()

    dprint("User query", q)

    # ── Step 1: Load conversation history ─────────────────────────────────────
    history: list[Turn] = memory.recent() if memory else []
    log.info(
        "PIPELINE START | question=%r | history_turns=%d | top_k=%d | threshold=%.2f",
        q, len(history), top_k, threshold,
    )

    # ── Step 2: Rewrite query ─────────────────────────────────────────────────
    rewritten_q = q
    was_rewritten = False

    if history:
        try:
            rw = _rewrite_query(q, history, api_key)
            rewritten_q = rw.rewritten
            was_rewritten = rw.was_rewritten
            if was_rewritten:
                log.info(
                    "REWRITTEN | original=%r | rewritten=%r | explanation=%s",
                    q, rewritten_q, rw.explanation or "(none)",
                )
                dprint("Rewritten query", rewritten_q)
            else:
                log.info("REWRITE SKIPPED | query is already self-contained")
        except Exception as exc:
            log.warning(
                "REWRITE FAILED (using original) | question=%r | error=%s", q, exc
            )
            rewritten_q = q
            was_rewritten = False

    # ── Step 3: Retrieve ───────────────────────────────────────────────────────
    try:
        chunks: list[dict] = retriever.retrieve(rewritten_q, top_k=top_k, threshold=threshold)
    except Exception as exc:
        reason = f"Retrieval failed: {exc}"
        log.error("RETRIEVAL ERROR | question=%r | error=%s", rewritten_q, exc, exc_info=True)
        return _make_result(
            "retrieval_error", q, rewritten_q, was_rewritten, [], None, reason, _ms(t0),
            user_message=f"Document retrieval failed: {exc}",
        )

    dprint("Chunks found", len(chunks))

    ids = [str(c.get("id", "unknown")) for c in chunks]
    valid_ids = set(ids)
    log.info("RETRIEVED | count=%d | ids=%s", len(chunks), ids)

    # ── Step 4a: Empty result ──────────────────────────────────────────────────
    if len(chunks) == 0:
        reason = f"No chunks retrieved above threshold {threshold:.2f}."
        log.warning("REJECTED | question=%r | reason='%s'", rewritten_q, reason)
        return _make_result(
            "insufficient_basis", q, rewritten_q, was_rewritten, [], None, reason, _ms(t0),
            user_message="No relevant legal provisions found in dataset.",
        )

    # ── Step 4b: Below minimum chunk count ────────────────────────────────────
    if len(chunks) < MIN_CHUNKS:
        reason = (
            f"Only {len(chunks)} chunk(s) retrieved above threshold {threshold:.2f} "
            f"-- minimum required is {MIN_CHUNKS}."
        )
        log.warning(
            "REJECTED | question=%r | retrieved=%d | required=%d | ids=%s",
            rewritten_q, len(chunks), MIN_CHUNKS, ids,
        )
        return _make_result(
            "insufficient_basis", q, rewritten_q, was_rewritten, chunks, None, reason, _ms(t0),
            user_message="Insufficient legal basis to provide a reliable answer.",
        )

    # ── Step 5: Citation engine ────────────────────────────────────────────────
    try:
        cited = generate_cited_answer(rewritten_q, chunks, api_key=api_key)
    except Exception as exc:
        reason = f"Citation engine failed: {exc}"
        log.error("CITATION ERROR | question=%r | error=%s", rewritten_q, exc, exc_info=True)
        return _make_result(
            "citation_error", q, rewritten_q, was_rewritten, chunks, None, reason, _ms(t0),
            user_message=f"Legal citation engine error: {exc}",
        )

    if cited.get("error") == "INSUFFICIENT_LEGAL_BASIS":
        reason = "Citation engine returned INSUFFICIENT_LEGAL_BASIS."
        log.warning("REJECTED by cite.py | question=%r | ids=%s", rewritten_q, ids)
        return _make_result(
            "insufficient_basis", q, rewritten_q, was_rewritten, chunks, None, reason, _ms(t0),
            user_message="Insufficient legal basis to provide a reliable answer.",
        )

    # ── Step 6: Policy enforcement ────────────────────────────────────────────
    try:
        enforce(cited, valid_ids)
    except PolicyViolation as pv:
        log.error(
            "POLICY VIOLATION | [%s] %s | question=%r",
            pv.rule, pv.detail, rewritten_q,
        )
        result = policy_error_result(q, pv, chunks, _ms(t0))
        result["user_message"] = f"Answer rejected due to policy violation: {pv.detail}"
        return result

    # ── Step 7: Save to memory ────────────────────────────────────────────────
    conclusion = cited.get("conclusion") or cited.get("analysis", "")
    if memory is not None:
        try:
            memory.add(q, conclusion)
            log.info("MEMORY SAVED | total_turns=%d", len(memory))
        except Exception as exc:
            log.warning("MEMORY SAVE FAILED | %s", exc)

    elapsed = _ms(t0)
    log.info(
        "PIPELINE OK | question=%r | rewritten=%r | chunks_used=%s | elapsed_ms=%d",
        q, rewritten_q, cited.get("citations", ids), elapsed,
    )
    return _make_result(
        "ok", q, rewritten_q, was_rewritten, chunks, cited, None, elapsed,
        user_message=conclusion,
    )


def _ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


# ── Convenience: build retriever + run in one call ────────────────────────────

def answer(
    question: str,
    chunks: list[dict[str, Any]],
    api_key: str,
    top_k: int = 5,
    threshold: float = DEFAULT_THRESHOLD,
    embed_model: str = DEFAULT_MODEL,
    memory: ConversationMemory | None = None,
) -> AnswerResult:
    """
    One-shot convenience function -- builds the retriever and runs the pipeline.

    Use ``run()`` directly when you have a pre-built LegalRetriever
    (e.g. in a web server that indexes the corpus once at startup).

    Parameters
    ----------
    question    : user's legal question
    chunks      : full corpus (list of dicts with id, text, source, article)
    api_key     : Anthropic API key
    top_k       : maximum chunks to retrieve
    threshold   : minimum score threshold
    embed_model : sentence-transformers model name
    memory      : optional ConversationMemory; pass None to skip memory

    Returns
    -------
    AnswerResult -- always a dict, never raises.
    """
    try:
        retriever = LegalRetriever(chunks, embed_model=embed_model)
    except Exception as exc:
        reason = f"Failed to build retriever: {exc}"
        log.error("RETRIEVER BUILD ERROR | %s", exc, exc_info=True)
        return _make_result(
            "error", question, question, False, [], None, reason, 0,
            user_message=f"Failed to initialise document index: {exc}",
        )

    return run(
        question, retriever, api_key,
        top_k=top_k, threshold=threshold, memory=memory,
    )


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, pprint, sys

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Set ANTHROPIC_API_KEY to run the smoke test.")
        sys.exit(1)

    configure_logging(level=logging.DEBUG)

    corpus = [
        {
            "id": "k1",
            "source": "Law No. 05/L-085 on Electricity",
            "article": "Article 5",
            "paragraph": "Paragraph 1",
            "text": (
                "KOSTT, as the transmission system operator, is responsible for the "
                "secure, reliable and efficient operation of the transmission network, "
                "ensuring non-discriminatory access to all users."
            ),
        },
        {
            "id": "k2",
            "source": "Law No. 05/L-085 on Electricity",
            "article": "Article 12",
            "paragraph": "Paragraph 3",
            "text": (
                "The transmission system operator shall publish connection terms and "
                "conditions, including technical and financial criteria, and shall not "
                "discriminate between users seeking access to the network."
            ),
        },
        {
            "id": "k3",
            "source": "ZRRE Network Code",
            "article": "Article 8",
            "paragraph": "Paragraph 2",
            "text": (
                "Connection requests must be processed within 30 days of receipt. "
                "Rejection must be justified in writing and may be appealed to the "
                "Energy Regulatory Office."
            ),
        },
    ]

    mem = ConversationMemory()   # in-process memory, no file

    print("\n" + "=" * 60)
    print("TEST A: First question (no history -- no rewriting)")
    print("=" * 60)
    r1 = answer(
        "Can KOSTT refuse a network connection without written justification?",
        corpus, api_key, threshold=0.2, memory=mem,
    )
    print(f"Status      : {r1['status']}")
    print(f"Rewritten   : {r1['rewritten_question']}")
    print(f"Was rewritten: {r1['was_rewritten']}")
    print(f"Elapsed     : {r1['elapsed_ms']} ms")
    if r1["answer"]:
        print(f"Conclusion  : {r1['answer'].get('conclusion')}")

    print("\n" + "=" * 60)
    print("TEST B: Follow-up with reference (memory has 1 turn)")
    print("=" * 60)
    r2 = answer(
        "Does that 30-day rule apply to emergency situations too?",
        corpus, api_key, threshold=0.2, memory=mem,
    )
    print(f"Status       : {r2['status']}")
    print(f"Original     : {r2['question']}")
    print(f"Rewritten    : {r2['rewritten_question']}")
    print(f"Was rewritten: {r2['was_rewritten']}")
    if r2["answer"]:
        print(f"Conclusion   : {r2['answer'].get('conclusion')}")

    print("\n" + "=" * 60)
    print("TEST C: Empty question (should be rejected)")
    print("=" * 60)
    r3 = answer("", corpus, api_key, memory=mem)
    print(f"Status : {r3['status']}")
    print(f"Reason : {r3['rejection_reason']}")

    print(f"\nMemory now holds {len(mem)} turn(s).")
