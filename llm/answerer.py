# llm/answerer.py

from typing import List, Dict
from llm.llm_client import LLMClient

llm = LLMClient()

SYSTEM_PROMPT = """You are an agricultural expert assistant.
Answer ONLY using the provided context.
If the answer is not present, say:
"Not found in the provided documents."
"""

def generate_answer(query: str, docs: List[Dict]) -> str:
    """
    Build final prompt and call LLM client.
    """

    if not docs:
        return "Not found in the provided documents."

    context_blocks = []

    for d in docs:
        text = d.get("text") or d.get("content", "")
        page = d.get("page_number", "unknown")
        context_blocks.append(f"[Page {page}]\n{text}")

    context = "\n\n".join(context_blocks)

    user_prompt = f"""
Context:
{context}

Question:
{query}

Give a precise, factual answer.
"""

    return llm.generate(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=400
    )
