from typing import List, Dict, Optional
import re
from llm.llm_client import LLMClient

# ---------------- CONFIG ----------------

MAX_CONTEXT_CHUNKS = 4
MAX_CHARS_PER_CHUNK = 700
MIN_KEYWORD_OVERLAP_RATIO = 0.3   # 🔥 proportional relevance
FALLBACK_CONFIDENCE_THRESHOLD = 0.5

SYSTEM_PROMPT = (
    "You are an agricultural expert assistant.\n\n"
    "Answer ONLY using the provided context.\n"
    "Do NOT add knowledge.\n"
    "Do NOT infer beyond the text.\n"
    "If the answer is not fully supported, say:\n"
    "\"Not found in the provided documents.\""
)

_llm: Optional[LLMClient] = None


def _get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm


# ---------------- TEXT HELPERS ----------------

def _clean_text(text: str) -> str:
    return " ".join(text.split()).strip()


def _extract_keywords(text: str) -> set:
    stopwords = {
        "what", "is", "are", "the", "about", "explain",
        "tell", "me", "give", "define", "how"
    }
    return {
        w for w in re.findall(r"[a-zA-Z]{4,}", text.lower())
        if w not in stopwords
    }


def _keyword_overlap_ratio(query: str, text: str) -> float:
    q_words = _extract_keywords(query)
    if not q_words:
        return 0.0
    t = text.lower()
    matched = sum(1 for w in q_words if w in t)
    return matched / len(q_words)


def _dedupe_chunks(docs: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for d in docs:
        cid = d.get("chunk_id")
        if cid and cid not in seen:
            seen.add(cid)
            out.append(d)
    return out


def _audit_answer(answer: str, context: str) -> float:
    """
    Measures how much of the answer is supported by context.
    """
    a_words = _extract_keywords(answer)
    if not a_words:
        return 0.0
    ctx = context.lower()
    supported = sum(1 for w in a_words if w in ctx)
    return supported / len(a_words)


# ---------------- MAIN ----------------

def generate_answer(
    query: str,
    docs: List[Dict],
    retrieval_diagnostics: Optional[Dict] = None
) -> Dict:
    """
    Returns:
    {
        answer: str
        confidence: float
        grounded: bool
        allow_fallback: bool
    }
    """

    # ---------- HARD FAIL ----------
    if not docs:
        return _fallback("no_docs")

    if retrieval_diagnostics:
        if retrieval_diagnostics.get("status") == "fail":
            return _fallback(retrieval_diagnostics.get("reason"))

    docs = _dedupe_chunks(docs)
    if not docs:
        return _fallback("empty_after_dedupe")

    # ---------- CONTEXT SELECTION ----------
    relevant_chunks = []
    relevance_scores = []

    for d in docs:
        text = _clean_text(d.get("text", ""))[:MAX_CHARS_PER_CHUNK]
        overlap = _keyword_overlap_ratio(query, text)

        if overlap >= MIN_KEYWORD_OVERLAP_RATIO:
            relevant_chunks.append(text)
            relevance_scores.append(overlap)

        if len(relevant_chunks) >= MAX_CONTEXT_CHUNKS:
            break

    if not relevant_chunks:
        return _fallback("no_relevant_context")

    context = "\n\n".join(relevant_chunks)

    # ---------- CONFIDENCE COMPONENTS ----------
    retrieval_strength = min(1.0, sum(relevance_scores) / len(relevance_scores))
    query_coverage = min(1.0, len(relevant_chunks) / MAX_CONTEXT_CHUNKS)

    # ---------- LLM ----------
    llm = _get_llm()
    answer = llm.generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:",
        temperature=0.0,
        max_tokens=220
    ).strip()

    if not answer or "not found in the provided documents" in answer.lower():
        return _fallback("llm_refused")

    answer_grounding = _audit_answer(answer, context)

    # ---------- FINAL CONFIDENCE ----------
    final_confidence = round(
        0.45 * retrieval_strength +
        0.35 * query_coverage +
        0.20 * answer_grounding,
        3
    )

    if final_confidence < FALLBACK_CONFIDENCE_THRESHOLD:
        return {
            "answer": "",
            "confidence": final_confidence,
            "grounded": False,
            "allow_fallback": True
        }

    return {
        "answer": answer,
        "confidence": final_confidence,
        "grounded": True,
        "allow_fallback": False
    }


def _fallback(reason: str) -> Dict:
    return {
        "answer": "",
        "confidence": 0.0,
        "grounded": False,
        "allow_fallback": True,
        "reason": reason
    }



