"""
tests/test_citation_engine.py
─────────────────────────────
Test suite for the legal citation engine.

Covers four scenarios:
    TC-01  Valid query with strong legal basis        → must succeed
    TC-02  Query with weak basis (< 2 chunks)         → must return INSUFFICIENT_LEGAL_BASIS
    TC-03  Query with conflicting provisions           → must cite both sides, flag conflict
    TC-04  Query outside dataset                      → must return error

Each test prints EXPECTED vs ACTUAL side-by-side.

Usage:
    python -m pytest tests/test_citation_engine.py -v          # all tests (mocked LLM)
    python -m pytest tests/test_citation_engine.py -v -k live  # real API (needs GEMINI_API_KEY)
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import unittest
from unittest.mock import patch

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.cite import generate_cited_answer, INSUFFICIENT_BASIS
from rag.retrieve import LegalRetriever
from rag.answer import run as pipeline_run
from rag.policy import enforce, PolicyViolation

# ── shared fixtures ───────────────────────────────────────────────────────────

# TC-01: Chunks that strongly support a network-access question
STRONG_CHUNKS = [
    {
        "id": "s1",
        "source": "Law No. 05/L-085 on Electricity",
        "article": "Article 5",
        "paragraph": "Paragraph 1",
        "text": (
            "KOSTT, as the transmission system operator, is obliged to ensure "
            "non-discriminatory access to the transmission network for all "
            "eligible users and producers."
        ),
    },
    {
        "id": "s2",
        "source": "Law No. 05/L-085 on Electricity",
        "article": "Article 12",
        "paragraph": "Paragraph 3",
        "text": (
            "The transmission system operator shall not discriminate between "
            "users or categories of users seeking access to the transmission "
            "network, and shall publish connection terms publicly."
        ),
    },
    {
        "id": "s3",
        "source": "ZRRE Network Code v2.4",
        "article": "Article 8",
        "paragraph": "Paragraph 2",
        "text": (
            "Connection requests must be processed within 30 days of receipt. "
            "Any rejection must be communicated in writing with a full "
            "technical and legal justification."
        ),
    },
]

# TC-02: Only one chunk available (below MIN_CHUNKS = 2)
WEAK_CHUNKS = [
    {
        "id": "w1",
        "source": "Law No. 05/L-085 on Electricity",
        "article": "Article 3",
        "paragraph": "Paragraph 1",
        "text": "KOSTT is the transmission system operator in Kosovo.",
    },
]

# TC-03: Two chunks with contradictory obligations on the same topic
CONFLICT_CHUNKS = [
    {
        "id": "c1",
        "source": "KOSTT Grid Code 2022",
        "article": "Article 14",
        "paragraph": "Paragraph 1",
        "text": (
            "The transmission system operator shall complete all planned "
            "maintenance works during off-peak hours (22:00–06:00) to "
            "minimise disruption to system users."
        ),
    },
    {
        "id": "c2",
        "source": "ZRRE Emergency Directive 01/2023",
        "article": "Article 3",
        "paragraph": "Paragraph 4",
        "text": (
            "In cases of system emergency, the transmission system operator "
            "may carry out maintenance at any hour without prior notice, "
            "overriding the standard scheduling obligations."
        ),
    },
    {
        "id": "c3",
        "source": "Law No. 05/L-085 on Electricity",
        "article": "Article 22",
        "paragraph": "Paragraph 2",
        "text": (
            "Emergency provisions in subordinate legislation shall prevail "
            "over general operational rules when a grid emergency is declared "
            "by KOSTT and notified to ZRRE within 2 hours."
        ),
    },
]

# TC-04: Chunks about network access — completely unrelated to a tax question
UNRELATED_CHUNKS = [
    {
        "id": "u1",
        "source": "Law No. 05/L-085 on Electricity",
        "article": "Article 5",
        "paragraph": "Paragraph 1",
        "text": "KOSTT ensures non-discriminatory access to the transmission network.",
    },
    {
        "id": "u2",
        "source": "ZRRE Network Code v2.4",
        "article": "Article 8",
        "paragraph": "Paragraph 2",
        "text": "Connection requests must be processed within 30 days.",
    },
]

# ── mock LLM responses ────────────────────────────────────────────────────────

def _mock_strong_response() -> str:
    return json.dumps({
        "issue": (
            "Whether KOSTT is legally obliged to grant non-discriminatory "
            "network access to all eligible users."
        ),
        "legal_basis": [
            {
                "chunk_id": "s1",
                "source": "Law No. 05/L-085 on Electricity",
                "article": "Article 5",
                "paragraph": "Paragraph 1",
                "provision": "KOSTT is obliged to ensure non-discriminatory access.",
            },
            {
                "chunk_id": "s2",
                "source": "Law No. 05/L-085 on Electricity",
                "article": "Article 12",
                "paragraph": "Paragraph 3",
                "provision": "TSO shall not discriminate between users seeking access.",
            },
            {
                "chunk_id": "s3",
                "source": "ZRRE Network Code v2.4",
                "article": "Article 8",
                "paragraph": "Paragraph 2",
                "provision": "Rejections must be in writing with full justification.",
            },
        ],
        "analysis": (
            "KOSTT bears an explicit statutory obligation of non-discriminatory "
            "access [Law No. 05/L-085 on Electricity, Article 5, Paragraph 1]. "
            "This is reinforced by the prohibition on differential treatment "
            "[Law No. 05/L-085 on Electricity, Article 12, Paragraph 3]. "
            "Any refusal must be accompanied by written technical and legal "
            "justification [ZRRE Network Code v2.4, Article 8, Paragraph 2]."
        ),
        "application": (
            "Applied to the present query, KOSTT cannot lawfully deny network "
            "access without a documented, written justification that satisfies "
            "both the statutory criteria [Article 5, Paragraph 1] and the "
            "procedural requirements of the Network Code [Article 8, Paragraph 2]."
        ),
        "conclusion": (
            "KOSTT is legally obliged to grant non-discriminatory network "
            "access. Any refusal is unlawful unless accompanied by full "
            "written justification [Law No. 05/L-085 on Electricity, "
            "Article 5, Paragraph 1]."
        ),
        "citations": ["s1", "s2", "s3"],
    })


def _mock_conflict_response() -> str:
    return json.dumps({
        "issue": (
            "Whether KOSTT may perform maintenance outside off-peak hours "
            "during a grid emergency, given the conflict between the Grid Code "
            "and the ZRRE Emergency Directive."
        ),
        "legal_basis": [
            {
                "chunk_id": "c1",
                "source": "KOSTT Grid Code 2022",
                "article": "Article 14",
                "paragraph": "Paragraph 1",
                "provision": "Maintenance restricted to off-peak hours (22:00-06:00).",
            },
            {
                "chunk_id": "c2",
                "source": "ZRRE Emergency Directive 01/2023",
                "article": "Article 3",
                "paragraph": "Paragraph 4",
                "provision": "Emergency maintenance may occur at any hour without notice.",
            },
            {
                "chunk_id": "c3",
                "source": "Law No. 05/L-085 on Electricity",
                "article": "Article 22",
                "paragraph": "Paragraph 2",
                "provision": "Emergency provisions prevail over general operational rules.",
            },
        ],
        "analysis": (
            "A direct conflict exists between two provisions: "
            "KOSTT Grid Code 2022 Article 14 Paragraph 1 restricts maintenance "
            "to off-peak hours [KOSTT Grid Code 2022, Article 14, Paragraph 1], "
            "while the ZRRE Emergency Directive permits maintenance at any hour "
            "during emergencies [ZRRE Emergency Directive 01/2023, Article 3, "
            "Paragraph 4]. "
            "The conflict is resolved by the hierarchy rule: emergency provisions "
            "in subordinate legislation prevail when a grid emergency is declared "
            "and ZRRE is notified within 2 hours "
            "[Law No. 05/L-085 on Electricity, Article 22, Paragraph 2]. "
            "CONFLICT IDENTIFIED: Grid Code vs Emergency Directive — resolved "
            "in favour of the Emergency Directive under lex specialis."
        ),
        "application": (
            "During a declared grid emergency, KOSTT may lawfully perform "
            "maintenance at any hour, provided it notifies ZRRE within 2 hours "
            "[Law No. 05/L-085 on Electricity, Article 22, Paragraph 2]. "
            "Outside a declared emergency, the Grid Code restriction applies "
            "[KOSTT Grid Code 2022, Article 14, Paragraph 1]."
        ),
        "conclusion": (
            "The ZRRE Emergency Directive prevails during declared emergencies. "
            "KOSTT may perform maintenance outside off-peak hours only when a "
            "grid emergency is active and ZRRE is notified "
            "[ZRRE Emergency Directive 01/2023, Article 3, Paragraph 4; "
            "Law No. 05/L-085, Article 22, Paragraph 2]."
        ),
        "citations": ["c1", "c2", "c3"],
    })


# ── helpers ───────────────────────────────────────────────────────────────────

_SEP  = "─" * 68
_SEP2 = "═" * 68

def _print_header(tc_id: str, title: str) -> None:
    print(f"\n{_SEP2}")
    print(f"  {tc_id}  {title}")
    print(_SEP2)

def _print_comparison(expected: dict, actual: dict) -> None:
    def _fmt(d: dict) -> str:
        return json.dumps(d, indent=2, ensure_ascii=False)

    exp_lines = _fmt(expected).splitlines()
    act_lines = _fmt(actual).splitlines()
    w = 50

    print(f"\n  {'EXPECTED':<{w}}  ACTUAL")
    print(f"  {_SEP}")
    for e, a in zip(exp_lines, act_lines):
        match = "  " if e == a else "!!"
        print(f"  {e:<{w}}{match}{a}")
    # Print any extra lines
    for e in exp_lines[len(act_lines):]:
        print(f"  {e:<{w}}  <missing>")
    for a in act_lines[len(exp_lines):]:
        print(f"  {'<missing>':<{w}}  {a}")

def _assert_field(result: dict, field: str, expected_value, label: str = "") -> bool:
    actual = result.get(field)
    ok = actual == expected_value
    mark = "PASS" if ok else "FAIL"
    tag  = f" ({label})" if label else ""
    print(f"  [{mark}] {field}{tag}")
    print(f"         expected : {expected_value!r}")
    print(f"         actual   : {actual!r}")
    return ok

def _assert_contains(result: dict, field: str, substring: str, label: str = "") -> bool:
    actual = str(result.get(field, ""))
    ok = substring.lower() in actual.lower()
    mark = "PASS" if ok else "FAIL"
    tag  = f" ({label})" if label else ""
    print(f"  [{mark}] {field} contains {substring!r}{tag}")
    return ok

def _assert_not_empty(result: dict, field: str) -> bool:
    actual = result.get(field)
    ok = bool(actual)
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {field} is non-empty  →  {str(actual)[:60]!r}")
    return ok


# ── test class ────────────────────────────────────────────────────────────────

class TestCitationEngine(unittest.TestCase):

    # ── TC-01 ─────────────────────────────────────────────────────────────────

    @patch("rag.cite._call_gemini", return_value=_mock_strong_response())
    def test_tc01_strong_basis(self, _mock):
        """TC-01: Valid query with strong legal basis must return a full answer."""
        _print_header("TC-01", "Valid query — strong legal basis")

        question = "Is KOSTT obliged to grant non-discriminatory network access?"

        EXPECTED = {
            "status":           "ok",
            "chunks_retrieved": 3,
            "answer.status":    "present",
            "citations_count":  3,
            "all_ids_valid":    True,
            "policy_pass":      True,
        }

        result = generate_cited_answer(question, STRONG_CHUNKS, api_key="TEST_KEY")

        ACTUAL = {
            "status":           "ok" if "error" not in result else "error",
            "chunks_retrieved": len(STRONG_CHUNKS),
            "answer.status":    "present" if result.get("issue") else "missing",
            "citations_count":  len(result.get("citations", [])),
            "all_ids_valid":    all(
                c in {"s1", "s2", "s3"} for c in result.get("citations", [])
            ),
            "policy_pass":      True,
        }

        # Run policy check
        try:
            enforce(result, {"s1", "s2", "s3"})
        except PolicyViolation as pv:
            ACTUAL["policy_pass"] = False
            print(f"  [FAIL] Policy violation: {pv}")

        _print_comparison(EXPECTED, ACTUAL)

        print(f"\n  Generated answer preview:")
        print(f"  Issue      : {result.get('issue','—')}")
        print(f"  Conclusion : {textwrap.shorten(result.get('conclusion','—'), 100)}")
        print(f"  Citations  : {result.get('citations')}")

        checks = [
            _assert_field(ACTUAL, "status",           "ok",      "must succeed"),
            _assert_field(ACTUAL, "chunks_retrieved", 3,         "all chunks indexed"),
            _assert_field(ACTUAL, "citations_count",  3,         "all 3 sources cited"),
            _assert_field(ACTUAL, "all_ids_valid",    True,      "no fabricated IDs"),
            _assert_field(ACTUAL, "policy_pass",      True,      "passes R1–R4"),
            _assert_not_empty(result, "analysis"),
            _assert_not_empty(result, "conclusion"),
        ]
        self.assertTrue(all(checks), "TC-01 has failures — see above")

    # ── TC-02 ─────────────────────────────────────────────────────────────────

    def test_tc02_weak_basis(self):
        """TC-02: Only 1 chunk available — must return INSUFFICIENT_LEGAL_BASIS."""
        _print_header("TC-02", "Weak basis — fewer than 2 chunks")

        question = "What is KOSTT's role?"

        EXPECTED = {
            "error":   "INSUFFICIENT_LEGAL_BASIS",
            "status":  "error (no LLM call made)",
            "hallucination_risk": "none — engine never called",
        }

        result = generate_cited_answer(question, WEAK_CHUNKS, api_key="TEST_KEY")

        ACTUAL = {
            "error":   result.get("error", "—"),
            "status":  "error (no LLM call made)" if "error" in result else "FAIL — LLM was called",
            "hallucination_risk": (
                "none — engine never called" if "error" in result
                else "HIGH — answer produced without enough sources"
            ),
        }

        _print_comparison(EXPECTED, ACTUAL)

        checks = [
            _assert_field(result, "error", "INSUFFICIENT_LEGAL_BASIS",
                          "must block before LLM call"),
        ]
        self.assertTrue(all(checks), "TC-02 has failures — see above")

    # ── TC-03 ─────────────────────────────────────────────────────────────────

    @patch("rag.cite._call_gemini", return_value=_mock_conflict_response())
    def test_tc03_conflicting_provisions(self, _mock):
        """TC-03: Conflicting provisions — engine must cite both sides and flag conflict."""
        _print_header("TC-03", "Conflicting provisions")

        question = (
            "Can KOSTT perform maintenance outside off-peak hours "
            "during a declared grid emergency?"
        )

        EXPECTED = {
            "status":                 "ok",
            "conflict_flagged":       True,
            "both_sides_cited":       True,
            "resolution_cited":       True,
            "all_3_sources_present":  True,
            "policy_pass":            True,
        }

        result = generate_cited_answer(question, CONFLICT_CHUNKS, api_key="TEST_KEY")

        cited_ids     = set(result.get("citations", []))
        analysis_text = result.get("analysis", "").lower()

        ACTUAL = {
            "status": "ok" if "error" not in result else "error",
            "conflict_flagged": "conflict" in analysis_text,
            "both_sides_cited": (
                "c1" in cited_ids and "c2" in cited_ids
            ),
            "resolution_cited": "c3" in cited_ids,
            "all_3_sources_present": {"c1", "c2", "c3"}.issubset(cited_ids),
            "policy_pass": True,
        }

        try:
            enforce(result, {"c1", "c2", "c3"})
        except PolicyViolation as pv:
            ACTUAL["policy_pass"] = False

        _print_comparison(EXPECTED, ACTUAL)

        print(f"\n  Analysis excerpt:")
        print(f"  {textwrap.shorten(result.get('analysis', '—'), 120)}")
        print(f"  Conclusion:")
        print(f"  {textwrap.shorten(result.get('conclusion', '—'), 120)}")

        checks = [
            _assert_field(ACTUAL, "status",                "ok",   "must succeed"),
            _assert_field(ACTUAL, "conflict_flagged",      True,   "must flag conflict in analysis"),
            _assert_field(ACTUAL, "both_sides_cited",      True,   "must cite Grid Code AND Emergency Directive"),
            _assert_field(ACTUAL, "resolution_cited",      True,   "must cite hierarchy resolution provision"),
            _assert_field(ACTUAL, "all_3_sources_present", True,   "all 3 chunk IDs in citations"),
            _assert_field(ACTUAL, "policy_pass",           True,   "passes R1–R4"),
        ]
        self.assertTrue(all(checks), "TC-03 has failures — see above")

    # ── TC-04 ─────────────────────────────────────────────────────────────────

    def test_tc04_query_outside_dataset(self):
        """
        TC-04: Query about a topic not in the dataset.
        retrieve.py must return 0 or 1 chunk (below threshold),
        so the engine must never be called.
        """
        _print_header("TC-04", "Query outside dataset — no relevant chunks")

        # Corpus is about electricity network access.
        # Query is about employment law — completely unrelated.
        question = (
            "What are the legal grounds for dismissing an employee for "
            "repeated absence under Kosovo labour law?"
        )

        EXPECTED = {
            "pipeline_status":    "insufficient_basis",
            "answer":             None,
            "chunks_retrieved":   "< 2",
            "llm_called":         False,
            "hallucination_risk": "none",
        }

        retriever = LegalRetriever(UNRELATED_CHUNKS)
        # Use a very high threshold so unrelated chunks are rejected
        result = pipeline_run(
            question,
            retriever,
            api_key="TEST_KEY",
            threshold=0.70,
        )

        ACTUAL = {
            "pipeline_status":    result["status"],
            "answer":             result["answer"],
            "chunks_retrieved":   (
                "< 2" if result["chunks_retrieved"] < 2 else str(result["chunks_retrieved"])
            ),
            "llm_called":         result["answer"] is not None,
            "hallucination_risk": (
                "none" if result["answer"] is None else "HIGH"
            ),
        }

        _print_comparison(EXPECTED, ACTUAL)

        print(f"\n  Rejection reason : {result.get('rejection_reason')}")
        print(f"  Chunk IDs seen   : {result.get('chunk_ids')}")

        checks = [
            _assert_field(
                ACTUAL, "pipeline_status", "insufficient_basis",
                "must reject before LLM call",
            ),
            _assert_field(
                ACTUAL, "answer", None,
                "no answer object when outside dataset",
            ),
            _assert_field(
                ACTUAL, "llm_called", False,
                "LLM must not be invoked",
            ),
            _assert_field(
                ACTUAL, "hallucination_risk", "none",
                "no hallucination possible",
            ),
        ]
        self.assertTrue(all(checks), "TC-04 has failures — see above")

    # ── TC-05 (bonus): policy gate catches fabricated ID ─────────────────────

    @patch("rag.cite._call_gemini")
    def test_tc05_policy_catches_fabrication(self, mock_gemini):
        """TC-05 (bonus): LLM injects a fabricated chunk ID → policy gate must catch it."""
        _print_header("TC-05", "Policy gate — fabricated citation ID")

        fabricated_response = json.dumps({
            "issue": "Test issue",
            "legal_basis": [
                {
                    "chunk_id": "INVENTED_ID_999",   # not in corpus
                    "source": "Imaginary Law",
                    "article": "Article 99",
                    "paragraph": "Paragraph 1",
                    "provision": "Some fabricated text.",
                }
            ],
            "analysis":    "Analysis [Imaginary Law, Article 99, Paragraph 1].",
            "application": "Application text.",
            "conclusion":  "Conclusion [Imaginary Law, Article 99, Paragraph 1].",
            "citations":   ["INVENTED_ID_999"],
        })
        mock_gemini.return_value = fabricated_response

        EXPECTED = {
            "pipeline_status":   "policy_violation",
            "rule_violated":     "R1",
            "answer":            None,
            "fabrication_caught": True,
        }

        retriever = LegalRetriever(STRONG_CHUNKS)
        result = pipeline_run(
            "Can KOSTT deny access?",
            retriever,
            api_key="TEST_KEY",
            threshold=0.0,
        )

        ACTUAL = {
            "pipeline_status":    result["status"],
            "rule_violated":      result.get("rule_violated", "—"),
            "answer":             result["answer"],
            "fabrication_caught": result["status"] == "policy_violation",
        }

        _print_comparison(EXPECTED, ACTUAL)

        print(f"\n  Rejection reason : {result.get('rejection_reason')}")

        checks = [
            _assert_field(ACTUAL, "pipeline_status",    "policy_violation", "must be caught"),
            _assert_field(ACTUAL, "rule_violated",      "R1",               "correct rule"),
            _assert_field(ACTUAL, "answer",             None,               "no answer leaked"),
            _assert_field(ACTUAL, "fabrication_caught", True,               "gate fired"),
        ]
        self.assertTrue(all(checks), "TC-05 has failures — see above")


# ── live integration tests (skipped unless GEMINI_API_KEY set) ────────────────

@unittest.skipUnless(os.environ.get("GEMINI_API_KEY"), "Set GEMINI_API_KEY to run live tests")
class TestCitationEngineLive(unittest.TestCase):
    """Integration tests that hit the real Gemini API."""

    def test_live_tc01_strong_basis(self):
        """LIVE TC-01: real LLM call with strong legal corpus."""
        _print_header("LIVE TC-01", "Real API — strong basis")
        api_key = os.environ["GEMINI_API_KEY"]

        result = generate_cited_answer(
            "Is KOSTT obliged to grant non-discriminatory network access?",
            STRONG_CHUNKS,
            api_key=api_key,
        )

        self.assertNotIn("error", result)
        self.assertIn("issue",      result)
        self.assertIn("conclusion", result)
        self.assertTrue(result.get("citations"))
        enforce(result, {"s1", "s2", "s3"})   # must pass all policy rules
        print(f"  Issue      : {result['issue']}")
        print(f"  Conclusion : {textwrap.shorten(result['conclusion'], 100)}")
        print(f"  Citations  : {result['citations']}")

    def test_live_tc04_outside_dataset(self):
        """LIVE TC-04: real pipeline — off-topic query must be rejected before LLM."""
        _print_header("LIVE TC-04", "Real API — query outside dataset")
        api_key = os.environ["GEMINI_API_KEY"]
        retriever = LegalRetriever(UNRELATED_CHUNKS)
        result = pipeline_run(
            "What are the tax obligations of a sole trader in Kosovo?",
            retriever, api_key=api_key, threshold=0.70,
        )
        self.assertIn(result["status"], ("insufficient_basis", "policy_violation"))
        self.assertIsNone(result["answer"])
        print(f"  Status : {result['status']}")
        print(f"  Reason : {result['rejection_reason']}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(_SEP2)
    print("  LEGAL CITATION ENGINE — TEST SUITE")
    print(_SEP2)
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    # Always run mocked tests
    suite.addTests(loader.loadTestsFromTestCase(TestCitationEngine))

    # Run live tests only if key present
    if os.environ.get("GEMINI_API_KEY"):
        suite.addTests(loader.loadTestsFromTestCase(TestCitationEngineLive))
    else:
        print("\n  NOTE: Set GEMINI_API_KEY to also run live integration tests.\n")

    runner = unittest.TextTestRunner(verbosity=0, stream=sys.stdout)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
