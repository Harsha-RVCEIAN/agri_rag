# llm/answerer.py

from typing import List, Dict
import re

from llm.llm_client import LLMClient


# ---------------- LLM INIT ----------------

llm = LLMClient()

SYSTEM_PROMPT = (
    "You are an agricultural expert assistant.\n"
    "\n"
    "Answer using ONLY the provided context.\n"
    "Do NOT add information that is not present in the context.\n"
    "\n"
    "If the context does not contain enough information, say:\n"
    "\"Not found in the provided documents.\""
)


# ---------------- CONFIG ----------------

MAX_CONTEXT_CHUNKS = 5
MAX_CHARS_PER_CHUNK = 800


# ---------------- HELPERS ----------------

def _clean_text(text: str) -> str:
    """
    Normalize OCR / PDF text and make list-like content explicit.
    """
    text = " ".join(text.split())

    # normalize common list markers
    text = text.replace("Key Features", "Key Features:\n")
    text = text.replace(" - ", "\n- ")

    return text


def _is_list_question(query: str) -> bool:
    """
    Detect questions that REQUIRE multi-point answers.
    """
    q = query.lower()
    return any(
        kw in q
        for kw in [
            "feature", "features",
            "benefit", "benefits",
            "advantages",
            "key points",
            "components"
        ]
    )


def _extract_bullets_from_text(text: str) -> List[str]:
    """
    Deterministically extract list items from text.
    NO LLM involved.
    """
    items: List[str] = []

    parts = re.split(r"\n-\s+|\n•\s+|- ", text)

    for p in parts:
        p = p.strip()
        if len(p.split()) >= 3:
            items.append(p)

    return items


# ---------------- MAIN ----------------

def generate_answer(query: str, docs: List[Dict]) -> str:
    """
    Hybrid answer generation:
    - Deterministic extraction for list questions
    - LLM used ONLY for single-value / explanatory answers
    """

    if not docs:
        return "Not found in the provided documents."

    # =========================================================
    # LIST / FEATURES QUESTIONS → DETERMINISTIC PATH
    # =========================================================
    if _is_list_question(query):
        collected_items: List[str] = []

        for d in docs:
            text = d.get("text") or d.get("content", "")
            if not text:
                continue

            text = _clean_text(text)
            items = _extract_bullets_from_text(text)
            collected_items.extend(items)

        # deduplicate while preserving order
        seen = set()
        unique_items = []
        for i in collected_items:
            if i not in seen:
                seen.add(i)
                unique_items.append(i)

        if unique_items:
            return (
                "The features include:\n"
                + "\n".join(f"- {i}" for i in unique_items)
            )

        return "Not found in the provided documents."

    # =========================================================
    # NON-LIST QUESTIONS → LLM PATH
    # =========================================================
    context_blocks = []

    for d in docs[:MAX_CONTEXT_CHUNKS]:
        text = d.get("text") or d.get("content", "")
        if not text:
            continue

        text = _clean_text(text)[:MAX_CHARS_PER_CHUNK]

        page = d.get("page", d.get("page_number", "unknown"))
        source = d.get("source", "unknown")

        context_blocks.append(
            f"[Source: {source}, Page: {page}]\n{text}"
        )

    if not context_blocks:
        return "Not found in the provided documents."

    context = "\n\n".join(context_blocks)

    user_prompt = (
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{query}\n\n"
        "Instructions:\n"
        "- Use ONLY the provided context.\n"
        "- If a numeric value is clearly stated, extract it exactly.\n"
        "- Do NOT guess.\n\n"
        "Answer:"
    )

    return llm.generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=300
    )
