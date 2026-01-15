from typing import List, Dict

# =========================================================
# CONFIG
# =========================================================

# NORMAL MODE (RAG)
MAX_CONTEXT_CHUNKS = 6
MAX_CONTEXT_CHARS = 6000
MAX_CHARS_PER_CHUNK = 900   # 🔑 hard per-chunk cap

# FAST MODE (definitions / policies)
FAST_MAX_CHUNKS = 2
FAST_MAX_CHARS = 2200
FAST_MAX_CHARS_PER_CHUNK = 700


class PromptBuilder:
    """
    Formatting-only component.

    Responsibilities:
    - Context selection (NO re-ranking)
    - Context trimming (hard caps)
    - Prompt safety (anti-hallucination)

    Explicitly does NOT:
    - Score chunks
    - Decide relevance
    - Handle refusals
    """

    # =====================================================
    # MAIN
    # =====================================================

    def build(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        fast_mode: bool = False,
    ) -> Dict:

        used: List[Dict] = []
        seen_ids = set()
        total_chars = 0

        if fast_mode:
            max_chunks = FAST_MAX_CHUNKS
            max_chars = FAST_MAX_CHARS
            max_per_chunk = FAST_MAX_CHARS_PER_CHUNK
        else:
            max_chunks = MAX_CONTEXT_CHUNKS
            max_chars = MAX_CONTEXT_CHARS
            max_per_chunk = MAX_CHARS_PER_CHUNK

        # -------------------------------------------------
        # CONTEXT SELECTION (ORDER PRESERVED)
        # -------------------------------------------------
        for c in retrieved_chunks:
            cid = c.get("chunk_id")
            if not cid or cid in seen_ids:
                continue

            raw_text = (c.get("text") or "").strip()
            if not raw_text:
                continue

            text = raw_text[:max_per_chunk]
            text_len = len(text)

            if total_chars + text_len > max_chars:
                break

            c_copy = dict(c)
            c_copy["text"] = text

            seen_ids.add(cid)
            used.append(c_copy)
            total_chars += text_len

            if len(used) >= max_chunks:
                break

        # -------------------------------------------------
        # EMPTY CONTEXT GUARD (CRITICAL)
        # -------------------------------------------------
        if not used:
            return {
                "system_prompt": self._system_prompt(),
                "user_prompt": (
                    "CONTEXT:\n\n"
                    "QUESTION:\n"
                    f"{query}\n\n"
                    "ANSWER:\n"
                    "Not found in the provided documents."
                ),
                "used_chunks": [],
                "stats": {
                    "chunks_used": 0,
                    "context_chars": 0,
                    "fast_mode": fast_mode,
                },
            }

        context = self._format_context(used)

        return {
            "system_prompt": self._system_prompt(),
            "user_prompt": self._user_prompt(query, context),
            "used_chunks": used,
            "stats": {
                "chunks_used": len(used),
                "context_chars": total_chars,
                "fast_mode": fast_mode,
            },
        }

    # =====================================================
    # PROMPTS
    # =====================================================

    def _system_prompt(self) -> str:
        return (
            "SYSTEM ROLE: Agricultural Information Assistant\n\n"
            "RULES:\n"
            "1. Answer ONLY from the provided CONTEXT.\n"
            "2. Do NOT use outside knowledge.\n"
            "3. Do NOT infer or assume missing information.\n"
            "4. If the answer is missing, reply exactly:\n"
            "   'Not found in the provided documents.'\n"
            "5. Use tables for numeric answers when available.\n"
            "6. Be precise and concise.\n"
            "7. Never stop at mid-sentence or mid-thought.\n"
            "8. if it is extending token limit , then pick till its previous sentence and display it. \n"
            "9. 'Not found in the provided documents.'\n"
        )

    def _user_prompt(self, query: str, context: str) -> str:
        return (
            "CONTEXT:\n"
            f"{context}\n\n"
            "QUESTION:\n"
            f"{query}\n\n"
            "ANSWER:"
        )

    # =====================================================
    # CONTEXT FORMAT
    # =====================================================

    def _format_context(self, chunks: List[Dict]) -> str:
        formatted: List[str] = []

        for i, c in enumerate(chunks, start=1):
            header = []

            if c.get("source"):
                header.append(f"Source: {c['source']}")
            if c.get("page") is not None:
                header.append(f"Page: {c['page']}")
            if c.get("content_type"):
                header.append(f"Type: {c['content_type']}")

            block = (
                f"[CONTEXT {i}]\n"
                f"{' | '.join(header)}\n"
                f"{c.get('text', '')}"
            )

            formatted.append(block)

        return "\n\n---\n\n".join(formatted)
