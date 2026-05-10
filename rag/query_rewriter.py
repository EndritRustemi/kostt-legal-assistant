"""
query_rewriter.py -- Conversational Query Rewriter

Rewrites an ambiguous follow-up question into a standalone legal question
by resolving references ("that article", "this obligation", "previous case")
using the last three conversation turns as context.

Public API
----------
    rewrite(question, history, api_key) -> RewriteResult

Usage
-----
    from rag.query_rewriter import rewrite, Turn

    history = [
        Turn(
            question="What are KOSTT's connection obligations?",
            answer="Under Article 12 of Law 05/L-085, KOSTT must publish "
                   "connection terms and process requests within 30 days.",
        ),
    ]
    result = rewrite("Does that apply to generation facilities too?", history, api_key)
    print(result.rewritten)
    # -> "Does the connection obligation under Article 12 of Law No. 05/L-085
    #     apply to generation facility connection requests?"
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import anthropic

# ── Constants ──────────────────────────────────────────────────────────────────

_MODEL            = "claude-haiku-4-5-20251001"
_MAX_HISTORY_TURNS = 3
_ANSWER_SNIPPET_LEN = 400   # characters of each answer shown to the LLM


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class Turn:
    """One round of conversation: a user question and the assistant answer."""
    question: str
    answer: str


@dataclass
class RewriteResult:
    """
    Result of a rewrite operation.

    Attributes
    ----------
    original      : the question as the user typed it
    rewritten     : the standalone, fully-contextualized question
    was_rewritten : True if any references were resolved or context was added
    explanation   : one-sentence description of what was changed (empty if
                    was_rewritten is False)
    """
    original: str
    rewritten: str
    was_rewritten: bool
    explanation: str = field(default="")


# ── Internal helpers ───────────────────────────────────────────────────────────

def _format_history(history: list[Turn]) -> str:
    """Render the last N turns as a numbered block for the prompt."""
    turns = history[-_MAX_HISTORY_TURNS:]
    if not turns:
        return "(no prior conversation)"

    lines: list[str] = []
    for i, t in enumerate(turns, 1):
        snippet = t.answer.strip()
        if len(snippet) > _ANSWER_SNIPPET_LEN:
            snippet = snippet[:_ANSWER_SNIPPET_LEN].rsplit(" ", 1)[0] + " [...]"
        lines.append(
            f"[Turn {i}]\n"
            f"  Q: {t.question.strip()}\n"
            f"  A: {snippet}"
        )
    return "\n\n".join(lines)


def _build_prompt(question: str, history: list[Turn]) -> str:
    history_block = _format_history(history)

    return f"""You are a legal query clarification assistant for KOSTT (Kosovo's \
electricity transmission and market operator). Your only task is to rewrite \
follow-up questions so they are self-contained and unambiguous.

CONVERSATION HISTORY (most recent {_MAX_HISTORY_TURNS} turns):
------------------------------------------------------------
{history_block}
------------------------------------------------------------

CURRENT QUESTION:
"{question}"

REWRITING RULES — follow all of them exactly:

1. RESOLVE REFERENCES — Replace every vague reference with the specific term \
it refers to based on the conversation history:
   - "that article" / "this article" / "the article" -> full article citation \
     (e.g. "Article 12 of Law No. 05/L-085")
   - "that obligation" / "this obligation" -> the specific legal obligation \
     (e.g. "the 30-day connection processing obligation")
   - "previous case" / "this case" -> the specific legal situation discussed
   - "it" / "they" / "this" / "that" -> the specific legal entity or provision
   - "the law" -> the specific law name and number
   - "the regulator" -> ZRRE (Energy Regulatory Office) if context confirms it
   - "the operator" -> KOSTT or the specific operator named in context

2. PRESERVE INTENT — Do not change what the user is actually asking. \
Clarify context only; do not answer the question.

3. NO NEW FACTS — Do not add legal information that does not appear in the \
conversation history. Do not cite articles or laws not mentioned in the history.

4. LANGUAGE — Respond in the same language as the current question \
(Albanian or English).

5. NO REWRITE NEEDED — If the question is already fully self-contained \
(no ambiguous pronouns or references), set "was_rewritten" to false and \
return the question unchanged in "rewritten".

6. CONCISENESS — The rewritten question must be a single clear legal question. \
Do not split it into bullet points or sub-questions.

OUTPUT — Return only a JSON object with exactly these keys. \
No markdown, no prose, no code fences:
{{
  "rewritten": "<the standalone legal question>",
  "was_rewritten": <true|false>,
  "explanation": "<one sentence: what references were resolved, or empty string if was_rewritten is false>"
}}
"""


def _call_claude(prompt: str, api_key: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=_MODEL,
        max_tokens=512,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _extract_json(raw: str) -> dict:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    return json.loads(text.strip())


def _validate_output(data: dict) -> None:
    required = {"rewritten", "was_rewritten"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"LLM response missing required keys: {missing}")
    if not isinstance(data["rewritten"], str) or not data["rewritten"].strip():
        raise ValueError("'rewritten' must be a non-empty string")
    if not isinstance(data["was_rewritten"], bool):
        raise ValueError("'was_rewritten' must be a boolean")


# ── Public API ─────────────────────────────────────────────────────────────────

def rewrite(
    question: str,
    history: list[Turn],
    api_key: str,
) -> RewriteResult:
    """
    Rewrite a follow-up question into a fully standalone legal question.

    Parameters
    ----------
    question : str
        The user's current question, which may contain references to earlier
        turns ("that article", "this obligation", etc.).
    history : list[Turn]
        Ordered list of prior conversation turns. Only the last three are used.
        Pass an empty list when there is no prior context.
    api_key : str
        Anthropic API key (starts with sk-ant-).

    Returns
    -------
    RewriteResult
        Always returns a result. If the question is already self-contained,
        ``was_rewritten`` is False and ``rewritten`` equals ``question``.

    Raises
    ------
    ValueError
        If ``api_key`` or ``question`` is empty.
    RuntimeError
        If the Claude API call fails or returns structurally invalid output.
    """
    if not api_key:
        raise ValueError("api_key is required")

    q = question.strip()
    if not q:
        raise ValueError("question cannot be empty")

    # If no history, the question cannot reference anything — return as-is.
    if not history:
        return RewriteResult(original=q, rewritten=q, was_rewritten=False)

    prompt = _build_prompt(q, history)

    try:
        raw = _call_claude(prompt, api_key)
    except anthropic.APIError as exc:
        raise RuntimeError(f"Claude API request failed: {exc}") from exc

    try:
        data = _extract_json(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(
            f"LLM returned invalid JSON.\nError: {exc}\nRaw (first 400 chars): {raw[:400]}"
        ) from exc

    try:
        _validate_output(data)
    except ValueError as exc:
        raise RuntimeError(f"LLM output failed validation: {exc}") from exc

    return RewriteResult(
        original=q,
        rewritten=data["rewritten"].strip(),
        was_rewritten=bool(data["was_rewritten"]),
        explanation=data.get("explanation", "").strip(),
    )


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Set ANTHROPIC_API_KEY to run the smoke test.")
        sys.exit(1)

    _history = [
        Turn(
            question="What are KOSTT's obligations regarding connection requests?",
            answer=(
                "Under Article 12, Paragraph 3 of Law No. 05/L-085 on Electricity, "
                "KOSTT must publish connection terms and conditions and must not "
                "discriminate between users. Under Article 8, Paragraph 2 of the "
                "ZRRE Network Code, connection requests must be processed within "
                "30 days and any rejection must be justified in writing."
            ),
        ),
        Turn(
            question="Can KOSTT charge different fees to different applicants?",
            answer=(
                "No. Article 12 of Law No. 05/L-085 requires non-discriminatory "
                "access. KOSTT may not apply differential pricing without objective "
                "justification approved by ZRRE."
            ),
        ),
    ]

    _cases = [
        (
            "Does that 30-day obligation apply to generation facilities too?",
            "Reference to time limit from earlier turn",
        ),
        (
            "What happens if they miss that deadline?",
            "Pronoun 'they' + 'that deadline' reference",
        ),
        (
            "Can KOSTT waive the non-discrimination requirement by contract?",
            "Self-contained question — no rewrite expected",
        ),
        (
            "What does the article say exactly?",
            "Vague 'the article' reference",
        ),
    ]

    print()
    print("=" * 65)
    print("query_rewriter.py -- smoke test")
    print("=" * 65)

    for q, label in _cases:
        print(f"\n[{label}]")
        print(f"  Original : {q}")
        result = rewrite(q, _history, api_key)
        print(f"  Rewritten: {result.rewritten}")
        print(f"  Changed  : {result.was_rewritten}")
        if result.explanation:
            print(f"  Why      : {result.explanation}")

    print()
    print("=" * 65)
    print("Smoke test complete.")
    print("=" * 65)
