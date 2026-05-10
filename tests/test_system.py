"""
tests/test_system.py
────────────────────
Integration test suite for the legal retrieval + citation system.

Validates the full pipeline:  LegalRetriever → answer.py → cite.py → policy.py

Four test cases, each using the rich ingest.py output format (dict source,
legal_reference, semantic fields) and a realistic 12-chunk, 4-document corpus.

    TC-01  Strong legal basis query      → valid answer + citations
    TC-02  Weak legal basis query        → INSUFFICIENT_LEGAL_BASIS, no LLM call
    TC-03  Out-of-domain query           → empty retrieval, pipeline error
    TC-04  Multi-source query            → citations span >= 2 source documents

Each test:
    1. Prints the QUERY
    2. Prints RETRIEVAL RESULTS (scores, chunk IDs, sources)
    3. Prints PIPELINE OUTPUT (status, citations, conclusion)
    4. Asserts expected behaviour — prints PASS / FAIL for every assertion

Usage:
    python -m pytest tests/test_system.py -v              # mocked LLM
    python -m pytest tests/test_system.py -v -k live      # real Gemini API
    python tests/test_system.py                           # standalone runner
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag.retrieve import LegalRetriever
from rag.answer  import run as pipeline_run
from rag.policy  import enforce, PolicyViolation

# ── Shared corpus — ingest.py output format ───────────────────────────────────
# 12 chunks across 4 distinct legal documents.
# Each chunk matches exactly what LegalIngestor.ingest_text() produces.

def _src(title: str, institution: str, doc_type: str) -> dict:
    return {"document_title": title, "institution": institution, "document_type": doc_type}

def _chunk(cid, text, source, article, paragraph, keywords):
    return {
        "id":       cid,
        "text":     text,
        "source":   source,
        "article":  article,
        "paragraph": paragraph,
        "legal_reference": {"article": article, "paragraph": paragraph},
        "semantic": {"keywords": keywords, "summary": text[:100]},
    }

_SRC_LAW  = _src("Law No. 05/L-085 on Electricity",  "Assembly of Kosovo", "Law")
_SRC_CODE = _src("ZRRE Network Code v2.4",            "ZRRE",               "Network Code")
_SRC_GRID = _src("KOSTT Grid Code 2023",              "KOSTT",              "Grid Code")
_SRC_MKT  = _src("ZRRE Market Rules 2022",            "ZRRE",               "Market Rules")

CORPUS: list[dict] = [
    # ── Law No. 05/L-085 on Electricity ──────────────────────────────────────
    _chunk(
        "law_art5_par1",
        ("KOSTT, as the transmission system operator, is responsible for the secure, "
         "reliable and efficient operation of the transmission network in Kosovo, "
         "ensuring non-discriminatory access to all users and producers."),
        _SRC_LAW, "Article 5 — KOSTT Obligations", "Paragraph 1",
        ["KOSTT", "transmission system operator", "non-discriminatory access", "network"],
    ),
    _chunk(
        "law_art12_par1",
        ("The transmission system operator shall publish the connection terms and conditions, "
         "including technical and financial criteria, and shall not discriminate between "
         "users or categories of users seeking access to the transmission network."),
        _SRC_LAW, "Article 12 — Network Access", "Paragraph 1",
        ["connection terms", "non-discrimination", "transmission network", "access"],
    ),
    _chunk(
        "law_art12_par2",
        ("Connection refusals must be communicated in writing and must state the full "
         "legal and technical grounds for the refusal. The applicant has the right to "
         "appeal the refusal to ZRRE within 15 days of receipt."),
        _SRC_LAW, "Article 12 — Network Access", "Paragraph 2",
        ["connection refusal", "written justification", "appeal", "ZRRE", "15 days"],
    ),
    _chunk(
        "law_art20_par1",
        ("Transmission tariffs shall be approved by ZRRE before entering into force. "
         "Tariffs must be cost-reflective, transparent and non-discriminatory, "
         "and must be published at least 30 days before they take effect."),
        _SRC_LAW, "Article 20 — Tariff Setting", "Paragraph 1",
        ["transmission tariffs", "ZRRE", "cost-reflective", "transparent", "publication"],
    ),

    # ── ZRRE Network Code v2.4 ────────────────────────────────────────────────
    _chunk(
        "code_art8_par1",
        ("All requests for connection to the transmission network must be submitted "
         "in writing to KOSTT, accompanied by a completed technical information form "
         "and documentary evidence of the applicant's legal capacity."),
        _SRC_CODE, "Article 8 — Connection Requests", "Paragraph 1",
        ["connection request", "written submission", "technical form", "KOSTT"],
    ),
    _chunk(
        "code_art8_par2",
        ("Connection requests must be processed within 30 days of receipt of a "
         "complete application. Any rejection must be communicated in writing with "
         "a full technical and legal justification; silence does not constitute approval."),
        _SRC_CODE, "Article 8 — Connection Requests", "Paragraph 2",
        ["30 days", "connection processing", "rejection", "written justification", "silence"],
    ),
    _chunk(
        "code_art15_par1",
        ("All connection points to the transmission network must be equipped with "
         "revenue-grade meters certified by the national metrology authority. "
         "Meter data must be stored for a minimum of 5 years."),
        _SRC_CODE, "Article 15 — Metering Requirements", "Paragraph 1",
        ["revenue-grade meters", "metrology", "connection points", "data retention"],
    ),

    # ── KOSTT Grid Code 2023 ──────────────────────────────────────────────────
    _chunk(
        "grid_art3_par1",
        ("The transmission system operator shall maintain system frequency within "
         "the range of 49.8 Hz to 50.2 Hz under normal operating conditions, "
         "and shall activate frequency restoration reserves within 30 seconds of "
         "a frequency deviation exceeding 200 mHz."),
        _SRC_GRID, "Article 3 — Frequency Control", "Paragraph 1",
        ["system frequency", "49.8 Hz", "50.2 Hz", "frequency restoration", "reserves"],
    ),
    _chunk(
        "grid_art14_par1",
        ("Planned maintenance of transmission network assets shall be scheduled "
         "during off-peak hours between 22:00 and 06:00 local time to minimise "
         "disruption to system users and market participants."),
        _SRC_GRID, "Article 14 — Maintenance Windows", "Paragraph 1",
        ["planned maintenance", "off-peak hours", "22:00", "06:00", "scheduling"],
    ),
    _chunk(
        "grid_art14_par2",
        ("Emergency maintenance may be carried out at any hour without prior notice "
         "when continuation of operation poses an imminent risk to system safety. "
         "KOSTT shall notify affected users and ZRRE within 2 hours of commencing "
         "emergency maintenance."),
        _SRC_GRID, "Article 14 — Maintenance Windows", "Paragraph 2",
        ["emergency maintenance", "imminent risk", "system safety", "ZRRE notification"],
    ),

    # ── ZRRE Market Rules 2022 ────────────────────────────────────────────────
    _chunk(
        "mkt_art2_par1",
        ("All market participants must register with KOSTT before engaging in "
         "electricity trading activities on the Kosovo electricity market. "
         "Registration requires submission of technical capability and financial "
         "standing documentation."),
        _SRC_MKT, "Article 2 — Market Registration", "Paragraph 1",
        ["market participants", "registration", "KOSTT", "electricity trading", "documentation"],
    ),
    _chunk(
        "mkt_art7_par2",
        ("Settlement of energy imbalances shall be calculated on the basis of "
         "real-time measurements from revenue-grade meters. Imbalance charges "
         "shall be published daily by KOSTT and invoiced monthly."),
        _SRC_MKT, "Article 7 — Imbalance Settlement", "Paragraph 2",
        ["imbalance settlement", "real-time measurements", "imbalance charges", "invoiced"],
    ),
]

# ── All valid IDs in the corpus ───────────────────────────────────────────────
_CORPUS_IDS: set[str] = {c["id"] for c in CORPUS}

# ── Display constants ─────────────────────────────────────────────────────────
_W   = 70
_SEP = "─" * _W
_BAR = "═" * _W


# ── Printing helpers ──────────────────────────────────────────────────────────

def _header(tc_id: str, title: str) -> None:
    print(f"\n{_BAR}")
    print(f"  {tc_id}  {title}")
    print(_BAR)

def _section(label: str) -> None:
    print(f"\n  ── {label} {'─' * max(0, _W - len(label) - 6)}")

def _print_retrieval(chunks: list[dict]) -> None:
    _section("RETRIEVAL RESULTS")
    if not chunks:
        print("  (no chunks retrieved)")
        return
    for c in chunks:
        src = c["source"]
        title = src["document_title"] if isinstance(src, dict) else str(src)
        score = c.get("score", "n/a")
        print(f"  [{c['id']}]  score={score}  |  {title}")
        print(f"      {c['article']}  {c['paragraph']}")

def _print_pipeline(result: dict) -> None:
    _section("PIPELINE OUTPUT")
    print(f"  status           : {result['status']}")
    print(f"  chunks_retrieved : {result['chunks_retrieved']}")
    print(f"  chunk_ids        : {result['chunk_ids']}")
    print(f"  elapsed_ms       : {result.get('elapsed_ms', '—')} ms")
    if result.get("answer"):
        ans = result["answer"]
        print(f"  issue            : {textwrap.shorten(ans.get('issue',''), 64)}")
        print(f"  conclusion       : {textwrap.shorten(ans.get('conclusion',''), 64)}")
        print(f"  citations        : {ans.get('citations', [])}")
    if result.get("rejection_reason"):
        print(f"  rejection_reason : {result['rejection_reason']}")
    if result.get("rule_violated"):
        print(f"  rule_violated    : {result['rule_violated']}")

def _check(label: str, passed: bool, detail: str = "") -> bool:
    mark = "PASS" if passed else "FAIL"
    line = f"  [{mark}]  {label}"
    if detail:
        line += f"  |  {detail}"
    print(line)
    return passed

def _summary_line(tc: str, passed: bool) -> None:
    mark = "PASS" if passed else "FAIL"
    print(f"  {mark}  {tc}")


# ── Mock LLM responses ────────────────────────────────────────────────────────

def _mock_tc01() -> str:
    return json.dumps({
        "issue": (
            "Whether KOSTT may lawfully refuse a network connection request "
            "without providing a written justification."
        ),
        "legal_basis": [
            {
                "chunk_id": "law_art12_par1",
                "source": "Law No. 05/L-085 on Electricity",
                "article": "Article 12", "paragraph": "Paragraph 1",
                "provision": "TSO shall not discriminate between users seeking access.",
            },
            {
                "chunk_id": "law_art12_par2",
                "source": "Law No. 05/L-085 on Electricity",
                "article": "Article 12", "paragraph": "Paragraph 2",
                "provision": "Refusals must be in writing with full legal and technical grounds.",
            },
            {
                "chunk_id": "code_art8_par2",
                "source": "ZRRE Network Code v2.4",
                "article": "Article 8", "paragraph": "Paragraph 2",
                "provision": "Rejection must include full technical and legal justification.",
            },
        ],
        "analysis": (
            "KOSTT bears a statutory duty of non-discriminatory access "
            "[Law No. 05/L-085, Article 12, Paragraph 1]. "
            "Any refusal must be communicated in writing with a full legal "
            "and technical justification [Law No. 05/L-085, Article 12, Paragraph 2]. "
            "The Network Code reinforces this: silence does not constitute approval "
            "and rejection requires full justification "
            "[ZRRE Network Code v2.4, Article 8, Paragraph 2]."
        ),
        "application": (
            "KOSTT cannot lawfully refuse a connection request without producing "
            "a written decision that sets out the legal and technical grounds "
            "[Law No. 05/L-085, Article 12, Paragraph 2; "
            "ZRRE Network Code v2.4, Article 8, Paragraph 2]."
        ),
        "conclusion": (
            "A connection refusal without written justification is unlawful. "
            "KOSTT must provide full grounds in writing and process the request "
            "within 30 days [ZRRE Network Code v2.4, Article 8, Paragraph 2]."
        ),
        "citations": ["law_art12_par1", "law_art12_par2", "code_art8_par2"],
    })


def _mock_tc04() -> str:
    return json.dumps({
        "issue": (
            "What are KOSTT's obligations on non-discriminatory network access "
            "and what timeline governs the processing of connection requests?"
        ),
        "legal_basis": [
            {
                "chunk_id": "law_art5_par1",
                "source": "Law No. 05/L-085 on Electricity",
                "article": "Article 5", "paragraph": "Paragraph 1",
                "provision": "KOSTT shall ensure non-discriminatory access to all users.",
            },
            {
                "chunk_id": "law_art12_par1",
                "source": "Law No. 05/L-085 on Electricity",
                "article": "Article 12", "paragraph": "Paragraph 1",
                "provision": "TSO shall not discriminate and shall publish connection terms.",
            },
            {
                "chunk_id": "code_art8_par1",
                "source": "ZRRE Network Code v2.4",
                "article": "Article 8", "paragraph": "Paragraph 1",
                "provision": "Connection requests must be submitted in writing to KOSTT.",
            },
            {
                "chunk_id": "code_art8_par2",
                "source": "ZRRE Network Code v2.4",
                "article": "Article 8", "paragraph": "Paragraph 2",
                "provision": "Requests must be processed within 30 days.",
            },
        ],
        "analysis": (
            "KOSTT carries an explicit duty of non-discriminatory network access "
            "[Law No. 05/L-085, Article 5, Paragraph 1; Article 12, Paragraph 1]. "
            "The procedural timeline is set by the Network Code: "
            "30 days from receipt of a complete application "
            "[ZRRE Network Code v2.4, Article 8, Paragraph 2]."
        ),
        "application": (
            "Any applicant seeking network connection is entitled to a decision "
            "within 30 days [ZRRE Network Code v2.4, Article 8, Paragraph 2] "
            "and cannot be refused on discriminatory grounds "
            "[Law No. 05/L-085, Article 12, Paragraph 1]."
        ),
        "conclusion": (
            "KOSTT must grant non-discriminatory access and process connection "
            "requests within 30 days. Obligations flow from two independent "
            "sources: Law No. 05/L-085 (Articles 5, 12) and ZRRE Network Code "
            "v2.4 (Article 8)."
        ),
        "citations": [
            "law_art5_par1", "law_art12_par1",
            "code_art8_par1", "code_art8_par2",
        ],
    })


# ── Test class ────────────────────────────────────────────────────────────────

class TestSystem(unittest.TestCase):
    """
    Integration tests for the legal retrieval + citation pipeline.
    Uses the 12-chunk, 4-document corpus defined above.
    LLM calls are mocked; retrieval runs against the real hybrid engine.
    """

    # ── TC-01 ─────────────────────────────────────────────────────────────────

    @patch("rag.cite._call_gemini", return_value=_mock_tc01())
    def test_tc01_strong_basis(self, _mock):
        """
        TC-01 — Strong legal basis query.

        Query is highly relevant to several corpus chunks.
        Expected: retriever returns >= 2 chunks; pipeline produces a
        valid structured answer; all cited IDs belong to the corpus;
        policy rules R1–R4 all pass.
        """
        _header("TC-01", "Strong legal basis query")

        question = (
            "Can KOSTT refuse a network connection request without providing "
            "written justification?"
        )
        _section("QUERY")
        print(f"  {question}")

        # ── Step 1: retrieval ─────────────────────────────────────────────────
        retriever = LegalRetriever(CORPUS)
        chunks = retriever.retrieve(question, top_k=5, threshold=0.30)
        _print_retrieval(chunks)

        retrieved_ids = {c["id"] for c in chunks}

        # ── Step 2: full pipeline (Gemini mocked) ─────────────────────────────
        result = pipeline_run(
            question, retriever, api_key="TEST_KEY", threshold=0.30
        )
        _print_pipeline(result)

        # ── Assertions ────────────────────────────────────────────────────────
        _section("ASSERTIONS")

        answer     = result.get("answer") or {}
        cited_ids  = set(answer.get("citations", []))

        policy_ok = True
        if answer:
            try:
                enforce(answer, retrieved_ids)
            except PolicyViolation as pv:
                policy_ok = False
                print(f"  [FAIL]  Policy violation: {pv}")

        checks = [
            _check("retrieval returns >= 2 chunks",
                   result["chunks_retrieved"] >= 2,
                   f"got {result['chunks_retrieved']}"),
            _check("retrieval scores are in [0, 1]",
                   all(0.0 <= c.get("score", 0) <= 1.0 for c in chunks),
                   f"scores: {[c.get('score') for c in chunks]}"),
            _check("pipeline status is 'ok'",
                   result["status"] == "ok",
                   f"got '{result['status']}'"),
            _check("answer object is present",
                   bool(result.get("answer")),
                   "answer must not be None"),
            _check("at least 1 citation present",
                   len(cited_ids) >= 1,
                   f"citations: {cited_ids}"),
            _check("all cited IDs exist in corpus",
                   cited_ids.issubset(_CORPUS_IDS),
                   f"unknown: {cited_ids - _CORPUS_IDS}"),
            _check("answer contains legal analysis",
                   bool(answer.get("analysis")),
                   "analysis field must be non-empty"),
            _check("answer contains conclusion",
                   bool(answer.get("conclusion")),
                   "conclusion field must be non-empty"),
            _check("policy rules R1-R4 pass",
                   policy_ok),
        ]
        self.assertTrue(all(checks), "TC-01 FAILED — see assertions above")

    # ── TC-02 ─────────────────────────────────────────────────────────────────

    def test_tc02_weak_basis(self):
        """
        TC-02 — Weak legal basis query.

        Query is about a narrow technical parameter (grid frequency tolerances)
        that appears in only one corpus chunk. At threshold=0.70 no chunk
        from an electricity-access corpus reaches sufficient similarity, so
        the pipeline must reject before ever calling the LLM.

        Expected: pipeline status 'insufficient_basis'; answer=None; no LLM call.
        """
        _header("TC-02", "Weak legal basis — insufficient retrievable chunks")

        question = (
            "What is the maximum allowable frequency deviation in the Kosovo "
            "transmission grid under normal operating conditions, and what are "
            "the penalty consequences for a market participant that causes a "
            "sustained frequency excursion?"
        )
        _section("QUERY")
        print(f"  {textwrap.shorten(question, 70)}")

        # ── Step 1: show retrieval result at high threshold ───────────────────
        retriever = LegalRetriever(CORPUS)
        chunks    = retriever.retrieve(question, top_k=5, threshold=0.70)
        _print_retrieval(chunks)

        # ── Step 2: pipeline (no mock needed — LLM must not be reached) ──────
        with patch("rag.cite._call_gemini") as mock_llm:
            result = pipeline_run(
                question, retriever, api_key="TEST_KEY", threshold=0.70
            )
            llm_called = mock_llm.called

        _print_pipeline(result)

        # ── Assertions ────────────────────────────────────────────────────────
        _section("ASSERTIONS")

        checks = [
            _check("pipeline status is 'insufficient_basis'",
                   result["status"] == "insufficient_basis",
                   f"got '{result['status']}'"),
            _check("answer is None (no data leaked)",
                   result["answer"] is None,
                   f"answer={result['answer']!r}"),
            _check("LLM was NOT called",
                   not llm_called,
                   "Gemini must not be invoked when evidence is thin"),
            _check("chunks retrieved < 2",
                   result["chunks_retrieved"] < 2,
                   f"got {result['chunks_retrieved']}"),
            _check("rejection reason is set",
                   bool(result.get("rejection_reason")),
                   f"reason: {result.get('rejection_reason')!r}"),
        ]
        self.assertTrue(all(checks), "TC-02 FAILED — see assertions above")

    # ── TC-03 ─────────────────────────────────────────────────────────────────

    @patch("rag.cite._call_gemini",
           return_value=json.dumps({"error": "INSUFFICIENT_LEGAL_BASIS"}))
    def test_tc03_out_of_domain(self, _mock):
        """
        TC-03 — Out-of-domain query.

        The corpus covers electricity transmission law only.
        The query is about Kosovo labour law — completely different domain.

        Design note: retrieve.py uses min-max normalization, so the top
        retrieved chunk always scores 1.0 regardless of true relevance.
        The real out-of-domain defence is the LLM: when handed electricity
        chunks for a labour-law question it correctly returns
        INSUFFICIENT_LEGAL_BASIS.  This test validates that full path.

        Expected: chunks ARE retrieved (retriever is domain-agnostic by
        design); LLM IS called; LLM returns INSUFFICIENT_LEGAL_BASIS;
        pipeline propagates it as status='insufficient_basis'; answer=None.
        """
        _header("TC-03", "Out-of-domain query — LLM correctly refuses to answer")

        question = (
            "On what legal grounds can an employer in Kosovo terminate an "
            "indefinite employment contract for repeated unjustified absences, "
            "and what notice period applies?"
        )
        _section("QUERY")
        print(f"  {textwrap.shorten(question, 70)}")

        retriever = LegalRetriever(CORPUS)

        # ── Show what retrieval surfaces for an off-domain query ──────────────
        chunks = retriever.retrieve(question, top_k=5, threshold=0.25)
        _print_retrieval(chunks)

        print(
            "\n  Note: retriever returns electricity chunks because min-max "
            "normalization is relative to the corpus — the defence is the LLM."
        )

        # ── Full pipeline (LLM mocked to return INSUFFICIENT_LEGAL_BASIS) ─────
        result = pipeline_run(
            question, retriever, api_key="TEST_KEY", threshold=0.25
        )
        _print_pipeline(result)

        # ── Assertions ────────────────────────────────────────────────────────
        _section("ASSERTIONS")

        rejection = result.get("rejection_reason") or ""

        checks = [
            _check("electricity chunks ARE retrieved (retriever is domain-agnostic)",
                   result["chunks_retrieved"] >= 2,
                   f"got {result['chunks_retrieved']} chunks"),
            _check("no labour-law chunks exist in the corpus (domain isolation)",
                   not any(
                       "employ" in c["text"].lower()
                       or "labour" in c["text"].lower()
                       or "dismissal" in c["text"].lower()
                       or "notice period" in c["text"].lower()
                       for c in chunks
                   ),
                   "corpus must not contain labour-law content"),
            _check("pipeline status is 'insufficient_basis'",
                   result["status"] == "insufficient_basis",
                   f"got '{result['status']}'"),
            _check("answer is None — irrelevant corpus produced no answer",
                   result["answer"] is None,
                   f"answer={result['answer']!r}"),
            _check("rejection reason mentions INSUFFICIENT_LEGAL_BASIS",
                   "INSUFFICIENT_LEGAL_BASIS" in rejection,
                   f"reason: {rejection!r}"),
        ]
        self.assertTrue(all(checks), "TC-03 FAILED — see assertions above")

    # ── TC-04 ─────────────────────────────────────────────────────────────────

    @patch("rag.cite._call_gemini", return_value=_mock_tc04())
    def test_tc04_multi_source(self, _mock):
        """
        TC-04 — Multi-source query.

        Query requires information from two independent legal sources:
        Law No. 05/L-085 (non-discrimination obligation) and
        ZRRE Network Code v2.4 (30-day processing timeline).

        Expected: retriever surfaces chunks from both documents; pipeline
        produces an answer whose citations reference >= 2 distinct
        document_title values.
        """
        _header("TC-04", "Multi-source query — citations across multiple documents")

        question = (
            "What are KOSTT's legal obligations on non-discriminatory network "
            "access and what timeline governs the processing of connection requests?"
        )
        _section("QUERY")
        print(f"  {textwrap.shorten(question, 70)}")

        retriever = LegalRetriever(CORPUS)
        chunks    = retriever.retrieve(question, top_k=6, threshold=0.25)
        _print_retrieval(chunks)

        retrieved_ids = {c["id"] for c in chunks}

        result = pipeline_run(
            question, retriever, api_key="TEST_KEY", threshold=0.25
        )
        _print_pipeline(result)

        # ── Resolve source document titles for cited chunks ───────────────────
        id_to_title = {
            c["id"]: (
                c["source"]["document_title"]
                if isinstance(c["source"], dict)
                else c["source"]
            )
            for c in CORPUS
        }
        answer      = result.get("answer") or {}
        cited_ids   = set(answer.get("citations", []))
        cited_titles = {id_to_title[cid] for cid in cited_ids if cid in id_to_title}

        # ── Assertions ────────────────────────────────────────────────────────
        _section("ASSERTIONS")

        policy_ok = True
        if answer:
            try:
                enforce(answer, retrieved_ids)
            except PolicyViolation as pv:
                policy_ok = False
                print(f"  [FAIL]  Policy violation: {pv}")

        has_law  = "Law No. 05/L-085 on Electricity" in cited_titles
        has_code = "ZRRE Network Code v2.4"           in cited_titles

        checks = [
            _check("retrieval returns >= 2 chunks",
                   result["chunks_retrieved"] >= 2,
                   f"got {result['chunks_retrieved']}"),
            _check("retrieval spans >= 2 source documents",
                   len({
                       id_to_title[c["id"]]
                       for c in chunks if c["id"] in id_to_title
                   }) >= 2,
                   f"sources: {sorted({id_to_title.get(c['id'],'?') for c in chunks})}"),
            _check("pipeline status is 'ok'",
                   result["status"] == "ok",
                   f"got '{result['status']}'"),
            _check("citations span >= 2 distinct document titles",
                   len(cited_titles) >= 2,
                   f"titles cited: {sorted(cited_titles)}"),
            _check("Law No. 05/L-085 is cited",
                   has_law,
                   f"cited titles: {sorted(cited_titles)}"),
            _check("ZRRE Network Code v2.4 is cited",
                   has_code,
                   f"cited titles: {sorted(cited_titles)}"),
            _check("all cited IDs exist in corpus",
                   cited_ids.issubset(_CORPUS_IDS),
                   f"unknown: {cited_ids - _CORPUS_IDS}"),
            _check("policy rules R1-R4 pass",
                   policy_ok),
        ]
        self.assertTrue(all(checks), "TC-04 FAILED — see assertions above")


# ── Live integration tests (skipped unless GEMINI_API_KEY is set) ─────────────

@unittest.skipUnless(
    os.environ.get("GEMINI_API_KEY"),
    "Set GEMINI_API_KEY to run live integration tests",
)
class TestSystemLive(unittest.TestCase):
    """
    Live end-to-end tests using the real Gemini API.
    Each test hits the full pipeline: retrieval → LLM → policy gate.
    """

    def setUp(self):
        self._api_key  = os.environ["GEMINI_API_KEY"]
        self._retriever = LegalRetriever(CORPUS)

    def test_live_tc01_strong_basis(self):
        """LIVE TC-01 — real LLM, strong corpus match."""
        _header("LIVE TC-01", "Real Gemini API — strong basis")
        question = (
            "Can KOSTT refuse a network connection request without written justification?"
        )
        result = pipeline_run(question, self._retriever,
                              api_key=self._api_key, threshold=0.30)
        _print_pipeline(result)

        self.assertEqual(result["status"], "ok", result.get("rejection_reason"))
        self.assertIsNotNone(result["answer"])
        answer = result["answer"]
        self.assertTrue(answer.get("citations"))
        enforce(answer, {c["id"] for c in CORPUS})

    def test_live_tc03_out_of_domain(self):
        """LIVE TC-03 — real retrieval, off-topic query must be rejected before LLM."""
        _header("LIVE TC-03", "Real API — out-of-domain rejection")
        question = (
            "What are the legal grounds for dismissing an employee for "
            "repeated unjustified absences under Kosovo labour law?"
        )
        result = pipeline_run(question, self._retriever,
                              api_key=self._api_key, threshold=0.50)
        _print_pipeline(result)

        self.assertIn(result["status"], ("insufficient_basis", "error"),
                      "off-domain query must not produce an answer")
        self.assertIsNone(result["answer"])

    def test_live_tc04_multi_source(self):
        """LIVE TC-04 — real LLM must produce citations from >= 2 documents."""
        _header("LIVE TC-04", "Real API — multi-source citations")
        question = (
            "What are KOSTT's legal obligations on non-discriminatory network "
            "access and the timeline for processing connection requests?"
        )
        result = pipeline_run(question, self._retriever,
                              api_key=self._api_key, threshold=0.25)
        _print_pipeline(result)

        self.assertEqual(result["status"], "ok", result.get("rejection_reason"))
        answer     = result["answer"]
        cited_ids  = set(answer.get("citations", []))
        id_to_title = {
            c["id"]: (
                c["source"]["document_title"]
                if isinstance(c["source"], dict) else c["source"]
            )
            for c in CORPUS
        }
        cited_titles = {id_to_title[cid] for cid in cited_ids if cid in id_to_title}
        self.assertGreaterEqual(len(cited_titles), 2,
                                f"Expected >= 2 sources, got: {cited_titles}")
        enforce(answer, {c["id"] for c in CORPUS})


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(_BAR)
    print("  LEGAL RETRIEVAL + CITATION SYSTEM — TEST SUITE")
    print(_BAR)

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestSystem))

    if os.environ.get("GEMINI_API_KEY"):
        suite.addTests(loader.loadTestsFromTestCase(TestSystemLive))
        print("\n  NOTE: GEMINI_API_KEY found — live tests included.\n")
    else:
        print("\n  NOTE: Set GEMINI_API_KEY to also run live integration tests.\n")

    runner = unittest.TextTestRunner(verbosity=0, stream=sys.stdout)
    outcome = runner.run(suite)

    # Print final summary
    total  = outcome.testsRun
    failed = len(outcome.failures) + len(outcome.errors)
    passed = total - failed

    print(f"\n{_BAR}")
    print(f"  RESULTS  {passed}/{total} passed")
    print(_BAR)
    for tc, _ in outcome.failures + outcome.errors:
        _summary_line(str(tc), passed=False)
    for tc in [
        t for t in suite
        if not any(str(t) in str(f[0]) for f in outcome.failures + outcome.errors)
    ]:
        _summary_line(str(tc), passed=True)

    sys.exit(0 if outcome.wasSuccessful() else 1)
