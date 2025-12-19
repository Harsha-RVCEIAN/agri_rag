from typing import List, Dict, Optional
import re

from llm.llm_client import LLMClient

# ---------------- CONFIG ----------------

MAX_CONTEXT_CHUNKS = 4
MAX_CHARS_PER_CHUNK = 700
MIN_KEYWORD_OVERLAP = 1  # 🔑 semantic relevance gate

SYSTEM_PROMPT = (
    "You are an agricultural expert assistant.\n\n"
    "Answer STRICTLY using the provided context.\n"
    "Summarize concisely.\n"
    "DO NOT repeat the same idea.\n"
    "DO NOT add external knowledge.\n"
    "DO NOT guess.\n\n"
    "If the context does not contain the answer, respond with:\n"
    "\"Not found in the provided documents.\""
)

# ---------------- LLM (LAZY SINGLETON) ----------------

_llm: Optional[LLMClient] = None


def _get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm


# ---------------- TEXT HELPERS ----------------

def _clean_text(text: str) -> str:
    text = " ".join(text.split())
    text = text.replace("Key Features", "Key Features:\n")
    text = text.replace(" - ", "\n- ")
    return text.strip()


def _split_sentences(text: str) -> List[str]:
    return re.split(r"(?<=[.!?])\s+", text)


def _dedupe_sentences(text: str) -> str:
    """
    Removes repeated sentences across chunks.
    Fixes repetition bug.
    """
    seen = set()
    output = []

    for s in _split_sentences(text):
        s_clean = s.strip().lower()
        if len(s_clean) < 10:
            continue
        if s_clean in seen:
            continue
        seen.add(s_clean)
        output.append(s.strip())

    return " ".join(output)


# ---------------- SEMANTIC RELEVANCE (CRITICAL) ----------------

def _extract_query_keywords(query: str) -> set:
    """
    Extract meaningful keywords from query.
    This prevents 'organic farming' → rice fertilizer leakage.
    """
    stopwords = {
        "what", "is", "are", "the", "about", "explain",
        "tell", "me", "give", "define", "how"
    }
    words = re.findall(r"[a-zA-Z]{4,}", query.lower())
    return {w for w in words if w not in stopwords}


def _is_relevant(query: str, text: str) -> bool:
    """
    HARD relevance gate.
    If this fails → fallback.
    """
    q_words = _extract_query_keywords(query)
    if not q_words:
        return False

    t = text.lower()
    overlap = sum(1 for w in q_words if w in t)
    return overlap >= MIN_KEYWORD_OVERLAP


# ---------------- LIST QUESTION HANDLING ----------------

def _is_list_question(query: str) -> bool:
    q = query.lower()
    return any(
        kw in q for kw in (
            "feature", "features",
            "benefit", "benefits",
            "advantages",
            "key points",
            "components"
        )
    )


def _extract_bullets(text: str) -> List[str]:
    parts = re.split(r"\n-\s+|\n•\s+|- ", text)
    bullets = []

    for p in parts:
        p = p.strip()
        if len(p.split()) >= 3:
            bullets.append(p)

    return bullets


# ---------------- CHUNK DEDUPE ----------------

def _dedupe_chunks(docs: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []

    for d in docs:
        cid = d.get("chunk_id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        unique.append(d)

    return unique


# ---------------- MAIN ANSWER FUNCTION ----------------

def generate_answer(
    query: str,
    docs: List[Dict],
    retrieval_diagnostics: Optional[Dict] = None
) -> Dict:
    """
    Returns:
    {
        answer: str
        grounded: bool
        allow_fallback: bool
    }
    """

    # ---------- NO DOCS ----------
    if not docs:
        return {
            "answer": "",
            "grounded": False,
            "allow_fallback": True
        }

    # ---------- RETRIEVER SAYS IRRELEVANT ----------
    if retrieval_diagnostics:
        reason = retrieval_diagnostics.get("reason")
        if reason in {"irrelevant_query", "weak_retrieval", "no_matches"}:
            return {
                "answer": "",
                "grounded": False,
                "allow_fallback": True
            }

    # ---------- DEDUPE ----------
    docs = _dedupe_chunks(docs)
    if not docs:
        return {
            "answer": "",
            "grounded": False,
            "allow_fallback": True
        }

    # ==================================================
    # LIST QUESTIONS (NO LLM GUESSING)
    # ==================================================
    if _is_list_question(query):
        bullets: List[str] = []

        for d in docs:
            text = d.get("text") or ""
            cleaned = _clean_text(text)
            if _is_relevant(query, cleaned):
                bullets.extend(_extract_bullets(cleaned))

        bullets = list(dict.fromkeys(bullets))  # dedupe, preserve order

        if bullets:
            return {
                "answer": "The key points are:\n" + "\n".join(f"- {b}" for b in bullets),
                "grounded": True,
                "allow_fallback": False
            }

        return {
            "answer": "",
            "grounded": False,
            "allow_fallback": True
        }

    # ==================================================
    # EXPLANATORY QUESTIONS (STRICT RAG)
    # ==================================================
    relevant_texts: List[str] = []

    for d in docs:
        text = d.get("text") or ""
        cleaned = _clean_text(text)[:MAX_CHARS_PER_CHUNK]

        if _is_relevant(query, cleaned):
            relevant_texts.append(cleaned)

        if len(relevant_texts) >= MAX_CONTEXT_CHUNKS:
            break

    if not relevant_texts:
        return {
            "answer": "",
            "grounded": False,
            "allow_fallback": True
        }

    context = _dedupe_sentences("\n\n".join(relevant_texts))

    user_prompt = (
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{query}\n\n"
        "Answer:"
    )

    llm = _get_llm()
    answer = llm.generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=250
    ).strip()

    if not answer or "not found in the provided documents" in answer.lower():
        return {
            "answer": "",
            "grounded": False,
            "allow_fallback": True
        }

    return {
        "answer": answer,
        "grounded": True,
        "allow_fallback": False
    }
