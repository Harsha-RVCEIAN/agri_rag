from typing import List, Dict, Optional
import re
from llm.llm_client import LLMClient

# =========================================================
# CONFIG
# =========================================================

MAX_CONTEXT_CHUNKS = 4
MAX_CHARS_PER_CHUNK = 700

MIN_KEYWORD_OVERLAP_RATIO = 0.25    # 🔑 relaxed but still strict
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


# =========================================================
# LLM SINGLETON
# =========================================================

def _get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm


# =========================================================
# TEXT HELPERS
# =========================================================

_STOPWORDS = {
    "what", "is", "are", "the", "about", "explain",
    "tell", "me", "give", "define", "how", "list",
    "describe", "details"
}


def _clean_text(text: str) -> str:
    return " ".join(text.split()).strip()


def _extract_keywords(text: str) -> set:
    return {
        w for w in re.findall(r"[a-zA-Z]{4,}", text.lower())
        if w not in _STOPWORDS
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
    Lexical grounding check.
    Conservative by design.
    """
    a_words = _extract_keywords(answer)
    if not a_words:
        return 0.0

    ctx = context.lower()
    supported = sum(1 for w in a_words if w in ctx)

    return supported / len(a_words)


# =========================================================
# MAIN
# =========================================================

def generate_answer(
    query: str,
    docs: List[Dict],
    retrieval_diagnostics: Optional[Dict] = None
) -> Dict:
    """
    Output contract (STABLE):

    {
        answer: str
        confidence: float
        grounded: bool
        allow_fallback: bool
        reason?: str
    }
    """

    # -----------------------------------------------------
    # HARD FAILS
    # -----------------------------------------------------
    if not docs:
        return _fallback("no_docs")

    if retrieval_diagnostics:
        if retrieval_diagnostics.get("status") == "fail":
            return _fallback(retrieval_diagnostics.get("reason", "retrieval_failed"))

    docs = _dedupe_chunks(docs)
    if not docs:
        return _fallback("empty_after_dedupe")

    # -----------------------------------------------------
    # CONTEXT SELECTION (LIGHTWEIGHT, NOT RAG)
    # -----------------------------------------------------
    selected_texts: List[str] = []
    relevance_scores: List[float] = []

    for d in docs:
        raw = _clean_text(d.get("text", ""))
        if not raw:
            continue

        text = raw[:MAX_CHARS_PER_CHUNK]
        overlap = _keyword_overlap_ratio(query, text)

        if overlap >= MIN_KEYWORD_OVERLAP_RATIO:
            selected_texts.append(text)
            relevance_scores.append(overlap)

        if len(selected_texts) >= MAX_CONTEXT_CHUNKS:
            break

    if not selected_texts:
        return _fallback("no_relevant_context")

    context = "\n\n".join(selected_texts)

    # -----------------------------------------------------
    # CONFIDENCE COMPONENTS (PRE-LLM)
    # -----------------------------------------------------
    retrieval_strength = min(1.0, sum(relevance_scores) / len(relevance_scores))
    query_coverage = min(1.0, len(selected_texts) / MAX_CONTEXT_CHUNKS)

    # -----------------------------------------------------
    # LLM CALL
    # -----------------------------------------------------
    llm = _get_llm()
    answer = llm.generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=(
            "CONTEXT:\n"
            f"{context}\n\n"
            "QUESTION:\n"
            f"{query}\n\n"
            "ANSWER:"
        ),
        temperature=0.0,
        max_tokens=220,
    ).strip()

    if not answer or "not found in the provided documents" in answer.lower():
        return _fallback("llm_refused")

    # -----------------------------------------------------
    # POST-LLM GROUNDING
    # -----------------------------------------------------
    answer_grounding = _audit_answer(answer, context)

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
            "allow_fallback": True,
            "reason": "low_final_confidence",
        }

    return {
        "answer": answer,
        "confidence": final_confidence,
        "grounded": True,
        "allow_fallback": False,
    }


# =========================================================
# FALLBACK
# =========================================================

def _fallback(reason: str) -> Dict:
    return {
        "answer": "",
        "confidence": 0.0,
        "grounded": False,
        "allow_fallback": True,
        "reason": reason,
    }
