from typing import List, Dict

# ---------------- CONFIG ----------------

MAX_CONTEXT_CHUNKS = 6
MAX_CONTEXT_CHARS = 8000


class PromptBuilder:
    """
    Formatting-only component.

    Responsibilities:
    - Context selection
    - Context structuring
    - Instruction hardening

    NO refusal logic.
    NO confidence logic.
    NO category decisions.
    """

    # ---------------- MAIN ----------------

    def build(self, query: str, retrieved_chunks: List[Dict]) -> Dict:
        used = []
        seen = set()
        total_chars = 0

        for c in retrieved_chunks:
            cid = c.get("chunk_id")
            if not cid or cid in seen:
                continue

            text = c.get("text", "").strip()
            if not text:
                continue

            if total_chars + len(text) > MAX_CONTEXT_CHARS:
                break

            seen.add(cid)
            used.append(c)
            total_chars += len(text)

            if len(used) >= MAX_CONTEXT_CHUNKS:
                break

        context = self._format_context(used)

        return {
            "system_prompt": self._system_prompt(),
            "user_prompt": self._user_prompt(query, context),
            "used_chunks": used,
        }

    # ---------------- PROMPT BUILDING ----------------

    def _system_prompt(self) -> str:
        """
        Hard constraints that survive prompt injection.
        """
        return (
            "SYSTEM ROLE: Agricultural Information Assistant\n\n"
            "NON-NEGOTIABLE RULES:\n"
            "1. Answer ONLY using the provided CONTEXT.\n"
            "2. Do NOT use prior knowledge.\n"
            "3. Do NOT infer, assume, or generalize.\n"
            "4. If the answer is NOT explicitly stated, reply EXACTLY:\n"
            "   'Not found in the provided documents.'\n"
            "5. For numeric information, rely ONLY on tables if present.\n"
            "6. Do NOT give recommendations unless explicitly written.\n"
            "7. Avoid definitive or prescriptive language.\n"
        )

    def _user_prompt(self, query: str, context: str) -> str:
        return (
            "CONTEXT (authoritative, limited):\n"
            f"{context}\n\n"
            "USER QUESTION:\n"
            f"{query}\n\n"
            "ANSWER RULES:\n"
            "- Use only information from the CONTEXT.\n"
            "- Keep the answer concise and factual.\n"
            "- Do NOT add explanations beyond the text.\n\n"
            "ANSWER:"
        )

    def _format_context(self, chunks: List[Dict]) -> str:
        """
        Structured context improves grounding and citation behavior.
        """
        formatted = []

        for i, c in enumerate(chunks, start=1):
            block = (
                f"[CONTEXT {i}]\n"
                f"Source: {c.get('source', 'unknown')}\n"
                f"Page: {c.get('page', 'N/A')}\n"
                f"Content-Type: {c.get('content_type', 'text')}\n"
                f"Content:\n{c.get('text','')}"
            )
            formatted.append(block)

        return "\n\n---\n\n".join(formatted)
