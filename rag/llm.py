"""
llm.py — Përgjigje me Claude (Anthropic).

Nëse dokumentet kanë përgjigje → RAG nga dokumentet.
Nëse jo           → Claude me njohuritë e tij (jo DuckDuckGo, i bllokuar në HF Spaces).
"""

from __future__ import annotations
import anthropic

MODEL               = "claude-sonnet-4-6"
RELEVANCE_THRESHOLD = 0.50   # >= 0.50 → përdor dokumentet

# Fraza që tregojnë se Claude nuk gjeti asgjë në kontekst
_NOT_FOUND = (
    "nuk gjendet", "nuk gjindet", "nuk përmend", "nuk përmendet",
    "nuk ofron", "nuk disponohet", "nuk është", "nuk ekziston",
    "nuk jepet", "nuk specifikohet", "nuk mund të gjej",
    "dokumentet e ngarkuara nuk",
    "not found", "not mentioned", "not specified",
)

SYSTEM_DOC = """Jeni asistent juridik i specializuar për legjislacionin e energjisë në Kosovë.
Punoni për KOSTT (Operatori i Sistemit, Transmisionit dhe Tregut).
Rregullat:
1. Përgjigjuni VETËM bazuar në dokumentet e dhëna si kontekst.
2. Citoni gjithmonë burimin: emrin e dokumentit dhe nenin/faqen.
3. Nëse informacioni nuk gjendet, thoni qartë: "Nuk gjendet në dokumentet e ngarkuara."
4. Stil juridik: formal, i saktë, pa paqartësi.
5. Përgjigjuni në gjuhën e pyetjes (shqip ose anglisht)."""

SYSTEM_GENERAL = """Jeni asistent juridik i specializuar për legjislacionin e energjisë në Kosovë.
Punoni për KOSTT. Dokumentet zyrtare nuk përmbajnë përgjigje për këtë pyetje.
Rregullat:
1. Përgjigjuni bazuar në njohuritë tuaja për legjislacionin e energjisë së Kosovës dhe BE-së.
2. Tregoni qartë: "Kjo përgjigje nuk bazohet në dokumentet e ngarkuara."
3. Rekomandoni verifikim me burime zyrtare para çdo veprimi juridik.
4. Stil juridik: formal, i saktë.
5. Përgjigjuni në gjuhën e pyetjes."""


def _call_claude(system: str, prompt: str, api_key: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        temperature=0.1,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Burimi: {c['source']} | Kategoria: {c['category']} | Faqja: {c['page']}\n"
            f"{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _is_not_found(answer: str) -> bool:
    """True only if Claude explicitly says it cannot find anything."""
    low = answer.lower()
    return any(phrase in low for phrase in _NOT_FOUND) and len(answer) < 600


def generate_answer(
    question: str,
    chunks: list[dict],
    api_key: str,
) -> tuple[str, list[dict], str]:
    """
    Returns (answer, sources, source_type).
    source_type: "documents" | "general" | "none"
    """
    if not chunks:
        return (
            "Nuk ka dokumente të ngarkuara. Ju lutem ngarkoni PDF-et e ligjeve.",
            [],
            "none",
        )

    best_score = max(c["score"] for c in chunks)

    # ── RAG nga dokumentet ────────────────────────────────────────────────────
    if best_score >= RELEVANCE_THRESHOLD:
        context = _build_context(chunks)
        prompt = (
            f"KONTEKST JURIDIK:\n{context}\n\n"
            f"PYETJA: {question}\n\n"
            f"Jepni përgjigje të saktë juridike me citime burimi."
        )
        answer = _call_claude(SYSTEM_DOC, prompt, api_key)

        if not _is_not_found(answer):
            sources = [
                {
                    "doc":      c["source"],
                    "category": c["category"],
                    "page":     c["page"],
                    "snippet":  c["snippet"],
                    "score":    c["score"],
                    "url":      "",
                }
                for c in chunks
            ]
            return answer, sources, "documents"

    # ── Fallback: njohuritë e Claude (jo DuckDuckGo) ─────────────────────────
    prompt = (
        f"PYETJA (dokumentet zyrtare nuk përmbajnë përgjigje të drejtpërdrejtë):\n"
        f"{question}\n\n"
        f"Përgjigjuni bazuar në njohuritë tuaja për legjislacionin e energjisë në Kosovë."
    )
    answer = _call_claude(SYSTEM_GENERAL, prompt, api_key)
    return answer, [], "general"
