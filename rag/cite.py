"""
cite.py — Strict Legal Citation Engine

Generates structured, fully-cited legal answers from retrieved chunks.
No statement is permitted without a traceable chunk reference.
"""

from __future__ import annotations

import json
import re
from typing import Any

import anthropic

# ── Constants ──────────────────────────────────────────────────────────────────

_MODEL    = "claude-haiku-4-5-20251001"
_MIN_CHUNKS = 2
_REQUIRED_FIELDS = frozenset(
    {"issue", "legal_basis", "analysis", "application", "conclusion", "citations"}
)
INSUFFICIENT_BASIS: dict[str, str] = {"error": "INSUFFICIENT_LEGAL_BASIS"}

# ── Internal helpers ───────────────────────────────────────────────────────────

def _fmt_chunk_block(chunks: list[dict]) -> str:
    lines = []
    for c in chunks:
        src = c["source"]
        if isinstance(src, dict):
            src = src.get("document_title", str(src))
        para = c.get("paragraph") or "—"
        lines.append(
            f"[ID: {c['id']}]\n"
            f"Source : {src}\n"
            f"Article: {c['article']}  |  Paragraph: {para}\n"
            f"Text   : {c['text'].strip()}\n"
        )
    return "\n".join(lines)


def _build_prompt(question: str, chunks: list[dict]) -> str:
    chunk_block = _fmt_chunk_block(chunks)
    ids = [str(c["id"]) for c in chunks]

    return f"""You are a strict legal analysis engine operating under the following ABSOLUTE rules.

═══════════════════════════ RULES ═══════════════════════════
1. SOURCE RESTRICTION — You may ONLY use the chunks listed under
   "AVAILABLE LEGAL CHUNKS". Do not reference any law, article,
   or principle that does not appear there.

2. MANDATORY INLINE CITATION — Every sentence that makes a legal
   claim MUST be immediately followed by a citation in this format:
       [Source Name, Article X, Paragraph Y]
   Omitting a citation is a critical error.

3. CHUNK ID TRACKING — The "citations" array must list every chunk
   ID you used. Valid IDs: {ids}

4. INSUFFICIENT DATA — If fewer than 2 chunks are relevant to the
   question, output ONLY:
       {{"error": "INSUFFICIENT_LEGAL_BASIS"}}

5. JSON ONLY — Output a single, valid JSON object. No markdown,
   no prose, no code fences before or after.
═════════════════════════════════════════════════════════════

AVAILABLE LEGAL CHUNKS:
────────────────────────────────────────────────────────────
{chunk_block}
────────────────────────────────────────────────────────────

QUESTION: {question}

OUTPUT SCHEMA (return exactly these keys):
{{
  "issue": "<one sentence identifying the legal issue>",
  "legal_basis": [
    {{
      "chunk_id": "<id>",
      "source": "<document name>",
      "article": "<article number>",
      "paragraph": "<paragraph number or null>",
      "provision": "<quoted or paraphrased key text>"
    }}
  ],
  "analysis": "<multi-sentence analysis; every legal claim must carry [Source, Article X, Paragraph Y]>",
  "application": "<how the cited law applies to the specific question; inline citations required>",
  "conclusion": "<clear legal conclusion with citation of the primary provision>",
  "citations": ["<chunk_id>", ...]
}}
"""


def _call_claude(prompt: str, api_key: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=_MODEL,
        max_tokens=4096,
        temperature=0.05,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _extract_json(raw: str) -> dict:
    """Strip markdown fences if present and parse JSON."""
    text = raw.strip()
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    return json.loads(text.strip())


def _validate_chunks(chunks: list[dict]) -> list[dict]:
    """Return only chunks that have the required fields."""
    required = {"id", "text", "source", "article"}
    return [c for c in chunks if required.issubset(c.keys()) and c["text"].strip()]


def _validate_output(result: dict, valid_ids: set[str]) -> dict:
    """
    Enforce output structure only — not citation integrity.
    ID validation is the exclusive responsibility of policy.py (Rule R1).
    Raises ValueError only for structural violations (missing fields, wrong types).
    """
    missing = _REQUIRED_FIELDS - result.keys()
    if missing:
        raise ValueError(f"LLM response is missing required fields: {missing}")

    if not isinstance(result["legal_basis"], list) or not result["legal_basis"]:
        raise ValueError("'legal_basis' must be a non-empty list")

    if not isinstance(result["citations"], list):
        raise ValueError("'citations' must be a list")

    # Normalise IDs to strings; leave IDs intact so policy.py can inspect them
    result["citations"] = [str(c) for c in result["citations"]]

    # Ensure every legal_basis entry declares a chunk_id (structural check only)
    for i, entry in enumerate(result["legal_basis"]):
        if "chunk_id" not in entry:
            raise ValueError(f"'legal_basis'[{i}] is missing required key 'chunk_id'")
        entry["chunk_id"] = str(entry["chunk_id"])   # normalise to str

    return result


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_cited_answer(
    user_question: str,
    retrieved_chunks: list[dict[str, Any]],
    api_key: str = "",
) -> dict[str, Any]:
    """
    Generate a strictly cited legal answer from the provided chunks.

    Parameters
    ----------
    user_question : str
        The legal question posed by the user.
    retrieved_chunks : list[dict]
        Each dict must contain:
            - id        : unique identifier (str or int)
            - text      : chunk content (str)
            - source    : document/law name (str)
            - article   : article number or label (str)
            - paragraph : paragraph reference (str, optional)
    api_key : str
        Anthropic API key (starts with sk-ant-).

    Returns
    -------
    dict
        On success:
            {
              "issue": str,
              "legal_basis": list[dict],
              "analysis": str,
              "application": str,
              "conclusion": str,
              "citations": list[str]
            }
        On insufficient data:
            {"error": "INSUFFICIENT_LEGAL_BASIS"}

    Raises
    ------
    ValueError
        If api_key or user_question is empty, or if the LLM response
        fails structural validation.
    RuntimeError
        If the Claude API call fails or returns malformed JSON.
    """
    if not api_key:
        raise ValueError("api_key is required")
    if not user_question or not user_question.strip():
        raise ValueError("user_question cannot be empty")

    valid_chunks = _validate_chunks(retrieved_chunks)

    if len(valid_chunks) < _MIN_CHUNKS:
        return dict(INSUFFICIENT_BASIS)

    valid_ids = {str(c["id"]) for c in valid_chunks}
    prompt = _build_prompt(user_question, valid_chunks)

    try:
        raw = _call_claude(prompt, api_key)
    except anthropic.APIError as exc:
        raise RuntimeError(f"Claude API request failed: {exc}") from exc

    try:
        result = _extract_json(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(
            f"LLM returned invalid JSON.\nError: {exc}\nRaw (first 500 chars): {raw[:500]}"
        ) from exc

    # LLM itself decided there was insufficient basis
    if "error" in result:
        return result

    result = _validate_output(result, valid_ids)
    return result


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, pprint

    _key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not _key:
        print("Set ANTHROPIC_API_KEY environment variable to run the smoke test.")
        raise SystemExit(1)

    _chunks = [
        {
            "id": "k1",
            "source": "Law No. 05/L-085 on Electricity",
            "article": "Article 5",
            "paragraph": "Paragraph 1",
            "text": (
                "KOSTT, as the transmission system operator, is responsible for the "
                "secure, reliable and efficient operation of the transmission network "
                "in Kosovo, ensuring non-discriminatory access to all users."
            ),
        },
        {
            "id": "k2",
            "source": "Law No. 05/L-085 on Electricity",
            "article": "Article 12",
            "paragraph": "Paragraph 3",
            "text": (
                "The transmission system operator shall publish the connection terms "
                "and conditions, including the technical and financial criteria, and "
                "shall not discriminate between users seeking access to the network."
            ),
        },
        {
            "id": "k3",
            "source": "ZRRE Network Code",
            "article": "Article 8",
            "paragraph": "Paragraph 2",
            "text": (
                "Connection requests must be processed within 30 days of receipt. "
                "Rejection of a connection request must be justified in writing and "
                "may be appealed to the Energy Regulatory Office."
            ),
        },
    ]

    _question = "Can KOSTT refuse a connection request without written justification?"

    print("Running cite.py smoke test...\n")
    _result = generate_cited_answer(_question, _chunks, api_key=_key)
    pprint.pprint(_result, width=100)
