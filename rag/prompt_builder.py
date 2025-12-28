from typing import List, Dict

# ---------------- CONFIG ----------------

MAX_CONTEXT_CHUNKS = 6
MAX_CONTEXT_CHARS = 8000

# FAST trimming (used implicitly by pipeline)
FAST_MAX_CHUNKS = 2
FAST_MAX_CHARS = 2500


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

        # conservative defaults (pipeline controls how many chunks it passes)
        max_chunks = MAX_CONTEXT_CHUNKS
        max_chars = MAX_CONTEXT_CHARS

        for c in retrieved_chunks:
            cid = c.get("chunk_id")
            if not cid or cid in seen:
                continue

            text = (c.get("text") or "").strip()
            if not text:
                continue

            text_len = len(text)
            if total_chars + text_len > max_chars:
                break

            seen.add(cid)
            used.append(c)
            total_chars += text_len

            if len(used) >= max_chunks:
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
            "CONTEXT:\n"
            f"{context}\n\n"
            "QUESTION:\n"
            f"{query}\n\n"
            "ANSWER:"
        )

    # ---------------- CONTEXT FORMAT ----------------

    def _format_context(self, chunks: List[Dict]) -> str:
        """
        Compact, structured context.
        Metadata kept minimal to reduce token load.
        """
        formatted = []

        for i, c in enumerate(chunks, start=1):
            source = c.get("source")
            page = c.get("page")
            ctype = c.get("content_type", "text")

            header_parts = []
            if source:
                header_parts.append(f"Source: {source}")
            if page is not None:
                header_parts.append(f"Page: {page}")
            if ctype:
                header_parts.append(f"Type: {ctype}")

            header = " | ".join(header_parts)

            block = (
                f"[CONTEXT {i}]\n"
                f"{header}\n"
                f"{c.get('text','')}"
            )

            formatted.append(block)

        return "\n\n---\n\n".join(formatted)
