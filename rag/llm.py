"""
Gjeneron përgjigje me Gemini Flash — thirrje direkte REST, pa SDK.
Fallback: nëse dokumentet nuk kanë përgjigje, kërkon në internet.
"""

import requests
from duckduckgo_search import DDGS

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"
RELEVANCE_THRESHOLD = 0.45  # nën këtë skor → kërko në internet

NOT_FOUND_PHRASES = [
    "nuk gjendet", "nuk gjindet", "nuk ka", "nuk përmend",
    "nuk përmendet", "nuk ofron", "nuk disponohet", "not found",
    "nuk është", "nuk ekziston", "nuk jepet", "nuk specifikohet",
    "dokumentet e ngarkuara nuk", "nuk mund të gjej",
]

SYSTEM_DOC = """Jeni asistent juridik i specializuar për legjislacionin e energjisë në Kosovë.
Punoni për KOSTT (Operatori i Sistemit, Transmisionit dhe Tregut).
Rregullat:
1. Përgjigjuni VETËM bazuar në dokumentet e dhëna si kontekst.
2. Citoni gjithmonë burimin: emrin e dokumentit dhe nenin/faqen.
3. Nëse informacioni nuk gjendet, thoni qartë: "Nuk gjendet në dokumentet e ngarkuara."
4. Stil juridik: formal, i saktë, pa paqartësi.
5. Përgjigjuni në gjuhën e pyetjes (shqip ose anglisht)."""

SYSTEM_WEB = """Jeni asistent juridik i specializuar për legjislacionin e energjisë në Kosovë.
Punoni për KOSTT. Keni marrë rezultate nga interneti si kontekst.
Rregullat:
1. Sintetizoni informacionin nga rezultatet e internetit.
2. Citoni URL-në e burimit për çdo informacion.
3. Tregoni qartë që informacioni vjen nga interneti, jo nga dokumentet zyrtare.
4. Rekomandoni verifikim me burime zyrtare para çdo veprimi juridik.
5. Stil juridik: formal, i saktë."""


def _call_gemini(system: str, prompt: str, api_key: str) -> str:
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048},
    }
    r = requests.post(GEMINI_URL, params={"key": api_key}, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def _web_search(query: str, max_results: int = 5) -> list[dict]:
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception:
        return []


def generate_answer(question: str, chunks: list[dict], api_key: str) -> tuple[str, list[dict], str]:
    """
    Kthon: (answer, sources, source_type)
    source_type: "documents" | "web" | "none"
    """
    if not chunks:
        return "Nuk ka dokumente të ngarkuara. Ju lutem ngarkoni PDF-et e ligjeve.", [], "none"

    best_score = max(c["score"] for c in chunks)

    # ── RAG nga dokumentet ────────────────────────────────────────────────────
    if best_score >= RELEVANCE_THRESHOLD:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[{i}] Burimi: {chunk['source']} | Kategoria: {chunk['category']} | Faqja: {chunk['page']}\n"
                f"{chunk['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)
        prompt = f"KONTEKST JURIDIK:\n{context}\n\nPYETJA: {question}\n\nJepni përgjigje të saktë juridike me citime burimi."
        answer = _call_gemini(SYSTEM_DOC, prompt, api_key)

        # Nëse LLM thotë "nuk gjendet" → kalo te interneti
        answer_lower = answer.lower()
        llm_not_found = any(phrase in answer_lower for phrase in NOT_FOUND_PHRASES)
        if not llm_not_found:
            sources = [{"doc": c["source"], "category": c["category"], "page": c["page"],
                        "snippet": c["snippet"], "score": c["score"], "url": ""} for c in chunks]
            return answer, sources, "documents"

    # ── Fallback: kërkim në internet ──────────────────────────────────────────
    search_query = f"legjislacion energji Kosovë KOSTT ZRRE {question}"
    web_results = _web_search(search_query)

    if not web_results:
        # Nëse edhe interneti dështon, përgjigje e drejtpërdrejtë nga Gemini
        prompt = f"PYETJA (pa kontekst dokumentesh): {question}\n\nPërgjigjuni bazuar në njohuritë tuaja për legjislacionin e energjisë."
        answer = _call_gemini(SYSTEM_WEB, prompt, api_key)
        return answer, [], "web"

    context_parts = []
    for i, r in enumerate(web_results, 1):
        context_parts.append(f"[{i}] {r.get('title','')}\nURL: {r.get('href','')}\n{r.get('body','')}")
    context = "\n\n---\n\n".join(context_parts)
    prompt = f"REZULTATE NGA INTERNETI:\n{context}\n\nPYETJA: {question}\n\nSintetizoni përgjigjen duke cituar URL-të."
    answer = _call_gemini(SYSTEM_WEB, prompt, api_key)

    web_sources = [{"doc": r.get("title", "Web"), "category": "Internet", "page": "-",
                    "snippet": r.get("body", "")[:120], "score": "-",
                    "url": r.get("href", "")} for r in web_results]
    return answer, web_sources, "web"
