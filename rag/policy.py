"""
policy.py — Global Legal AI Policy

These four rules govern EVERY module in this system.
No module may return a legal answer that violates them.

    Rule 1 — No fabricated citations
    Rule 2 — No answer without source
    Rule 3 — Error over guessing
    Rule 4 — Always preserve legal traceability
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("legal.policy")

# ── Rule registry (single source of truth) ────────────────────────────────────

RULES: dict[str, str] = {
    "R1": (
        "Never fabricate legal citations. "
        "Every citation must reference a chunk ID present in the provided corpus."
    ),
    "R2": (
        "Never answer without source. "
        "Every legal claim must be backed by at least one corpus chunk."
    ),
    "R3": (
        "Prefer returning error over guessing. "
        "When evidence is insufficient, return a structured error — never a partial answer."
    ),
    "R4": (
        "Always preserve legal traceability. "
        "Every answer must carry chunk_ids, citations, and legal_basis traceable to the corpus."
    ),
}

RULES_TEXT = "\n".join(f"  [{k}] {v}" for k, v in RULES.items())


# ── Policy violation ───────────────────────────────────────────────────────────

class PolicyViolation(Exception):
    """
    Raised when a module output violates one of the global legal AI rules.

    Attributes
    ----------
    rule   : rule code, e.g. "R1"
    detail : human-readable explanation of what was violated
    """

    def __init__(self, rule: str, detail: str) -> None:
        if rule not in RULES:
            raise ValueError(f"Unknown rule code: {rule!r}. Valid codes: {list(RULES)}")
        self.rule   = rule
        self.detail = detail
        super().__init__(f"[{rule}] {detail}")

    @property
    def rule_text(self) -> str:
        return RULES[self.rule]


# ── Individual rule checkers ───────────────────────────────────────────────────

def check_no_fabrication(result: dict[str, Any], valid_ids: set[str]) -> None:
    """
    Rule 1 — All cited chunk IDs must exist in the provided corpus.

    Checks both the top-level ``citations`` list and each entry in ``legal_basis``.
    """
    # Top-level citations list
    cited = {str(c) for c in result.get("citations", [])}
    fabricated = cited - valid_ids
    if fabricated:
        raise PolicyViolation(
            "R1",
            f"citations list contains IDs not present in corpus: {sorted(fabricated)}",
        )

    # Each legal_basis entry
    for i, entry in enumerate(result.get("legal_basis", [])):
        cid = str(entry.get("chunk_id", ""))
        if cid not in valid_ids:
            raise PolicyViolation(
                "R1",
                f"legal_basis[{i}] references unknown chunk_id={cid!r}",
            )


def check_no_answer_without_source(result: dict[str, Any]) -> None:
    """
    Rule 2 — Answer must have at least one source in both legal_basis and citations.
    """
    if not result.get("legal_basis"):
        raise PolicyViolation("R2", "'legal_basis' is empty — answer has no source.")
    if not result.get("citations"):
        raise PolicyViolation("R2", "'citations' is empty — answer has no source.")


def check_traceability(result: dict[str, Any]) -> None:
    """
    Rule 4 — All traceability fields must be present and non-empty.

    Required: issue, legal_basis, analysis, application, conclusion, citations.
    """
    required = ("issue", "legal_basis", "analysis", "application", "conclusion", "citations")
    missing  = [f for f in required if not result.get(f)]
    if missing:
        raise PolicyViolation(
            "R4",
            f"Answer is missing required traceability fields: {missing}",
        )

    # citations and legal_basis must be lists
    if not isinstance(result.get("legal_basis"), list):
        raise PolicyViolation("R4", "'legal_basis' must be a list.")
    if not isinstance(result.get("citations"), list):
        raise PolicyViolation("R4", "'citations' must be a list.")


# ── Master enforcement function ────────────────────────────────────────────────

def enforce(
    result: dict[str, Any],
    valid_ids: set[str],
    *,
    skip_rules: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run all policy checks on a citation-engine answer.

    Parameters
    ----------
    result     : output dict from ``cite.generate_cited_answer()``
    valid_ids  : set of chunk IDs that were actually passed to the engine
    skip_rules : optional list of rule codes to skip (e.g. ["R4"] for unit tests)

    Returns
    -------
    The same ``result`` dict, unmodified, if all rules pass.

    Raises
    ------
    PolicyViolation
        On the first rule that is broken.  The caller (answer.py) is
        responsible for converting this into a structured error result.
    """
    skip = set(skip_rules or [])

    if "R2" not in skip:
        check_no_answer_without_source(result)      # fastest check first

    if "R4" not in skip:
        check_traceability(result)

    if "R1" not in skip:
        check_no_fabrication(result, valid_ids)     # needs valid_ids, so last

    log.debug("POLICY OK | rules_checked=%s | citations=%s",
              [r for r in RULES if r not in skip],
              result.get("citations"))
    return result


# ── Convenience: build an error result that is policy-safe ────────────────────

def policy_error_result(
    question: str,
    violation: PolicyViolation,
    chunks: list[dict],
    elapsed_ms: int,
) -> dict[str, Any]:
    """
    Build a structured error dict from a PolicyViolation.
    Always safe to return to the caller — never contains fabricated data.
    """
    return {
        "status":           "policy_violation",
        "question":         question,
        "chunks_retrieved": len(chunks),
        "chunk_ids":        [str(c.get("id", "?")) for c in chunks],
        "answer":           None,
        "rejection_reason": f"{violation.rule}: {violation.detail}",
        "rule_violated":    violation.rule,
        "rule_text":        violation.rule_text,
        "elapsed_ms":       elapsed_ms,
    }


# ── Introspection ──────────────────────────────────────────────────────────────

def print_rules() -> None:
    """Print all active policy rules to stdout."""
    print("GLOBAL LEGAL AI POLICY")
    print("=" * 60)
    print(RULES_TEXT)
    print("=" * 60)


if __name__ == "__main__":
    print_rules()
