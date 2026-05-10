"""
query_expander.py -- Legal Query Expander

Expands a user query into 3-5 semantically equivalent variants to improve
retrieval recall. Each variant rephrases or expands the original using:

  - Albanian/English abbreviation resolution (BRE, ZRRE, KOSTT, ...)
  - Domain synonym expansion (jo-balance -> imbalance settlement)
  - Bilingual legal terminology (Albanian terms + English equivalents)
  - Relevant legal concepts implied by the query

The original query is always returned as the first element of the list.
No new legal facts are introduced; only meaning is expanded.

Public API
----------
    expand(query, api_key) -> ExpansionResult

Usage
-----
    from rag.query_expander import expand

    result = expand("Cilat jane detyrimet e BRE-ve per jo-balancin?", api_key)
    for q in result.queries:
        print(q)
    # -> "Cilat jane detyrimet e BRE-ve per jo-balancin?"
    # -> "What are the obligations of renewable energy sources for imbalance?"
    # -> "renewable energy producers balancing responsibility imbalance settlement"
    # -> "BRE obligations imbalance allocation settlement rules Kosovo"
    # -> ...
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import anthropic

# ── Constants ──────────────────────────────────────────────────────────────────

_MODEL       = "claude-haiku-4-5-20251001"
_MIN_QUERIES = 3
_MAX_QUERIES = 5

# Abbreviation glossary injected verbatim into the prompt.
# Add entries here as the domain grows; no code changes needed.
_GLOSSARY = """
ALBANIAN ABBREVIATIONS -> FULL TERMS -> ENGLISH EQUIVALENTS
------------------------------------------------------------
BRE      -> burimet e ripërtëritshme të energjisë -> renewable energy sources (RES)
KOSTT    -> Operatori i Sistemit, Transmisionit dhe Tregut -> Transmission System Operator (TSO)
ZRRE     -> Zyra e Rregullatorit të Energjisë -> Energy Regulatory Office (ERO)
OSHEE    -> Operatori i Shpërndarjes dhe Furnizimit me Energji Elektrike -> Distribution System Operator (DSO)
KESCO    -> Kosovo Electricity Supply Company -> supplier of last resort
BPP      -> furnizues balancues -> Balancing Power Provider
OTC      -> treg jashtë bursës -> over-the-counter market
PPA      -> kontratë për blerje energjie -> Power Purchase Agreement
CAPEX    -> shpenzime kapitale -> capital expenditure
OPEX     -> shpenzime operative -> operational expenditure

ALBANIAN TERMS -> ENGLISH EQUIVALENTS -> LEGAL SYNONYMS
------------------------------------------------------------
jo-balancim / jo-balancë   -> imbalance -> balancing deviation -> imbalance settlement
skema mbështetëse           -> support scheme -> support mechanism -> feed-in tariff / feed-in premium
tarifë mbështetëse          -> support tariff -> incentive tariff -> premium price
çmim referencë              -> reference price -> market reference price
detyrimi i balancimit       -> balancing responsibility -> balancing obligation
shmangie e balancimit       -> balancing deviation -> schedule deviation
vendosja e jo-balancit      -> imbalance allocation -> imbalance settlement -> imbalance charge
detyrim lidhës              -> connection obligation -> grid connection obligation
aksesi i palëve të treta    -> third-party access (TPA) -> non-discriminatory access
operatori i transmisionit   -> transmission system operator (TSO)
operatori i shpërndarjes    -> distribution system operator (DSO)
licencë operimi             -> operating licence -> market licence
ankand kapacitetesh         -> capacity auction -> capacity allocation
kapacitet i disponueshëm    -> available capacity -> available transmission capacity (ATC)
rregullator i energjisë     -> energy regulator -> regulatory authority
tarifë e transmetimit       -> transmission tariff -> transmission use-of-system charge (TUOS)
ndarje e rrjetit            -> grid unbundling -> ownership unbundling -> functional unbundling
planifikimi i rrjetit       -> network planning -> transmission development plan
lidhja në rrjet             -> network connection -> grid connection -> grid access
"""


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class ExpansionResult:
    """
    Result of a query expansion operation.

    Attributes
    ----------
    original       : the query exactly as the user typed it
    queries        : list of queries to retrieve with; original is always first
    terms_expanded : human-readable list of expansions that were applied
                     (empty if the query needed no expansion)
    """
    original: str
    queries: list[str]
    terms_expanded: list[str] = field(default_factory=list)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _build_prompt(query: str) -> str:
    return f"""You are a legal information retrieval specialist for the Kosovo \
electricity sector. Your task is to expand a user query into {_MIN_QUERIES}-{_MAX_QUERIES} \
distinct search queries that will improve document retrieval coverage.

DOMAIN GLOSSARY (use this to expand abbreviations and terms):
{_GLOSSARY}

ORIGINAL QUERY:
"{query}"

EXPANSION RULES — follow all of them exactly:

1. FIRST ITEM — The first query in your list must be the original query, \
   unchanged.

2. EXPAND ABBREVIATIONS — If the query contains any abbreviation from the \
   glossary (BRE, ZRRE, KOSTT, etc.), produce a variant that spells it out \
   fully in Albanian and another in English.

3. EXPAND DOMAIN TERMS — If the query uses a term that has synonyms in the \
   glossary (jo-balancim, skema mbështetëse, etc.), produce variants that use \
   the synonyms and legal equivalents.

4. ADD RELATED LEGAL CONCEPTS — Produce at least one variant that uses the \
   specific legal terminology a lawyer or regulator would use when drafting \
   the relevant provision (e.g. "imbalance settlement rules", \
   "balancing responsibility obligation", "renewable energy support scheme").

5. BILINGUAL — Include at least one Albanian-language variant and at least \
   one English-language variant in your output.

6. NO NEW FACTS — Do not introduce legal facts, article numbers, or \
   obligations that are not implied by the original query.

7. COUNT — Return between {_MIN_QUERIES} and {_MAX_QUERIES} queries total \
   (including the original). Never return fewer than {_MIN_QUERIES}.

8. TERMS EXPANDED — List each expansion you applied as a short phrase \
   (e.g. "BRE -> renewable energy sources"). If no expansion was needed, \
   return an empty list.

OUTPUT — Return only a JSON object. No markdown, no prose, no code fences:
{{
  "queries": [
    "<original query>",
    "<expanded variant 1>",
    "<expanded variant 2>",
    ...
  ],
  "terms_expanded": [
    "<term> -> <expansion>",
    ...
  ]
}}
"""


def _call_claude(prompt: str, api_key: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=_MODEL,
        max_tokens=1024,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _extract_json(raw: str) -> dict:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    return json.loads(text.strip())


def _validate_output(data: dict, original: str) -> None:
    if "queries" not in data or not isinstance(data["queries"], list):
        raise ValueError("'queries' must be a list")
    data["queries"] = [q for q in data["queries"] if isinstance(q, str) and q.strip()]
    if len(data["queries"]) < _MIN_QUERIES:
        raise ValueError(
            f"Expected at least {_MIN_QUERIES} queries, got {len(data['queries'])}"
        )
    # Enforce the original as first item regardless of what the LLM returned
    if not data["queries"] or data["queries"][0].strip() != original:
        data["queries"] = [original] + [
            q for q in data["queries"] if q.strip() != original
        ]
    # Cap at max
    data["queries"] = data["queries"][:_MAX_QUERIES]
    # Normalise terms_expanded to a flat list of strings
    te = data.get("terms_expanded", [])
    data["terms_expanded"] = [str(t) for t in te if str(t).strip()] if isinstance(te, list) else []


# ── Public API ─────────────────────────────────────────────────────────────────

def expand(query: str, api_key: str) -> ExpansionResult:
    """
    Expand a legal query into 3-5 semantically equivalent retrieval variants.

    Parameters
    ----------
    query   : the user's legal question or search query
    api_key : Anthropic API key (starts with sk-ant-)

    Returns
    -------
    ExpansionResult
        ``queries`` always contains the original as the first item.

    Raises
    ------
    ValueError
        If ``api_key`` or ``query`` is empty.
    RuntimeError
        If the Claude API call fails or returns structurally invalid output.

    Notes
    -----
    The function never invents legal facts. It only rephrases and expands
    terminology using the built-in glossary and domain knowledge.
    """
    if not api_key:
        raise ValueError("api_key is required")

    q = query.strip()
    if not q:
        raise ValueError("query cannot be empty")

    prompt = _build_prompt(q)

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
        _validate_output(data, q)
    except ValueError as exc:
        raise RuntimeError(f"LLM output failed validation: {exc}") from exc

    return ExpansionResult(
        original=q,
        queries=data["queries"],
        terms_expanded=data["terms_expanded"],
    )


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Set ANTHROPIC_API_KEY to run the smoke test.")
        sys.exit(1)

    _cases = [
        "Cilat jane detyrimet e BRE-ve per jo-balancin?",
        "What support schemes exist for renewable producers under ZRRE regulations?",
        "A mund te refuzoje KOSTT nje kerkese per lidhje ne rrjet?",
        "network connection obligations transmission operator Kosovo",
        "skema mbështetëse për prodhuesit e BRE-ve",
    ]

    print()
    print("=" * 65)
    print("query_expander.py -- smoke test")
    print("=" * 65)

    for query in _cases:
        print(f"\nOriginal : {query}")
        result = expand(query, api_key)
        print(f"Expanded ({len(result.queries)} queries):")
        for i, q in enumerate(result.queries):
            prefix = "  [original]" if i == 0 else f"  [{i}]      "
            print(f"{prefix} {q}")
        if result.terms_expanded:
            print(f"Terms    : {' | '.join(result.terms_expanded)}")

    print()
    print("=" * 65)
    print("Smoke test complete.")
    print("=" * 65)
