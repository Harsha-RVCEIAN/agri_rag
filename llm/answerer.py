from typing import List, Dict, Optional
import re

from llm.llm_client import LLMClient


# ---------------- CONFIG ----------------

MAX_CONTEXT_CHUNKS = 5
MAX_CHARS_PER_CHUNK = 800


SYSTEM_PROMPT = (
    "You are an agricultural expert assistant.\n\n"
    "Answer using ONLY the provided context.\n"
    "Do NOT add information that is not present in the context.\n\n"
    "If the context does not contain enough information, say:\n"
    "\"Not found in the provided documents.\""
)


# ---------------- LLM (LAZY) ----------------

_llm: Optional[LLMClient] = None


def _get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm


# ---------------- HELPERS ----------------

def _clean_text(text: str) -> str:
    text = " ".join(text.split())
    text = text.replace("Key Features", "Key Features:\n")
    text = text.replace(" - ", "\n- ")
    return text


def _is_list_question(query: str) -> bool:
    q = query.lower()
    return any(
        kw in q
        for kw in (
            "feature", "features",
            "benefit", "benefits",
            "advantages",
            "key points",
            "components"
        )
    )


def _extract_bullets(text: str) -> List[str]:
    items: List[str] = []
    parts = re.split(r"\n-\s+|\n•\s+|- ", text)

    for p in parts:
        p = p.strip()
        if len(p.split()) >= 3:
            items.append(p)

    return items


# ---------------- MAIN ----------------

def generate_answer(
    query: str,
    docs: List[Dict],
    retrieval_diagnostics: Optional[Dict] = None
) -> Dict:
    """
    Generates a RAG-grounded answer ONLY.

    Returns structured output so the caller
    can decide whether to fallback to an external LLM.
    """

    # ---------- SAFETY GATE ----------
    if not docs:
        return {
            "answer": "Not found in the provided documents.",
            "grounded": False,
            "allow_fallback": True
        }

    if retrieval_diagnostics and retrieval_diagnostics.get("weak_retrieval"):
        return {
            "answer": "Not found in the provided documents.",
            "grounded": False,
            "allow_fallback": True
        }

    # =========================================================
    # LIST / FEATURES QUESTIONS → DETERMINISTIC
    # =========================================================
    if _is_list_question(query):
        collected: List[str] = []

        for d in docs:
            text = d.get("text") or d.get("content", "")
            if not text:
                continue

            text = _clean_text(text)
            collected.extend(_extract_bullets(text))

        # dedupe, preserve order
        seen = set()
        final = []
        for i in collected:
            if i not in seen:
                seen.add(i)
                final.append(i)

        if final:
            return {
                "answer": "The features include:\n" + "\n".join(f"- {i}" for i in final),
                "grounded": True,
                "allow_fallback": False
            }

        return {
            "answer": "Not found in the provided documents.",
            "grounded": False,
            "allow_fallback": True
        }

    # =========================================================
    # EXPLANATORY QUESTIONS → LLM (GROUND-LOCKED)
    # =========================================================
    context_blocks = []

    for d in docs[:MAX_CONTEXT_CHUNKS]:
        text = d.get("text") or d.get("content", "")
        if not text:
            continue

        text = _clean_text(text)[:MAX_CHARS_PER_CHUNK]
        source = d.get("source", "unknown")
        page = d.get("page", "unknown")

        context_blocks.append(
            f"[Source: {source}, Page: {page}]\n{text}"
        )

    if not context_blocks:
        return {
            "answer": "Not found in the provided documents.",
            "grounded": False,
            "allow_fallback": True
        }

    context = "\n\n".join(context_blocks)

    user_prompt = (
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{query}\n\n"
        "Instructions:\n"
        "- Use ONLY the provided context.\n"
        "- Extract facts verbatim when possible.\n"
        "- Do NOT guess.\n\n"
        "Answer:"
    )

    llm = _get_llm()
    answer = llm.generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=300
    )

    return {
        "answer": answer.strip(),
        "grounded": True,
        "allow_fallback": False
    }
