from typing import List, Dict


# ---------------- CONFIG ----------------

MAX_CONTEXT_CHUNKS = 6
MAX_CONTEXT_CHARS = 8000  # hard safety limit


# ---------------- PROMPT BUILDER ----------------

class PromptBuilder:
    """
    Responsible for:
    - Selecting & structuring retrieved chunks
    - Enforcing grounding rules
    - Building a strict, non-creative RAG prompt
    """

    # ---------------- CONTEXT SELECTION ----------------

    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []

        for c in chunks:
            cid = c.get("chunk_id")
            if cid and cid not in seen:
                seen.add(cid)
                unique.append(c)

        return unique

    def _truncate_context(self, chunks: List[Dict]) -> List[Dict]:
        total_chars = 0
        selected = []

        for c in chunks:
            text = c.get("text", "")
            text_len = len(text)

            if total_chars + text_len > MAX_CONTEXT_CHARS:
                break

            selected.append(c)
            total_chars += text_len

        return selected

    def _select_context(self, retrieved_chunks: List[Dict]) -> List[Dict]:
        if not retrieved_chunks:
            return []

        chunks = self._deduplicate_chunks(retrieved_chunks)
        chunks = chunks[:MAX_CONTEXT_CHUNKS]
        chunks = self._truncate_context(chunks)

        return chunks

    # ---------------- CONTEXT FORMATTING ----------------

    def _format_context(self, chunks: List[Dict]) -> str:
        """
        Format chunks into a structured, traceable context block.
        Explicitly surface confidence and content type.
        """
        formatted = []

        for idx, c in enumerate(chunks, start=1):
            block = (
                f"[CONTEXT {idx}]\n"
                f"Source: {c.get('source', 'unknown')}\n"
                f"Page: {c.get('page', 'N/A')}\n"
                f"Content-Type: {c.get('content_type', 'text')}\n"
                f"Confidence: {c.get('confidence', 'unknown')}\n"
                f"Chunk-ID: {c.get('chunk_id', 'N/A')}\n"
                f"Content:\n{c.get('text', '')}\n"
            )
            formatted.append(block)

        return "\n---\n".join(formatted)

    # ---------------- SYSTEM RULES ----------------

    def _system_rules(self) -> str:
        """
        Non-negotiable grounding & priority rules.
        """
        return (
            "SYSTEM ROLE: Agriculture Document Assistant\n\n"
            "INSTRUCTION PRIORITY (HIGHEST → LOWEST):\n"
            "1. System rules\n"
            "2. Developer rules\n"
            "3. User question\n"
            "4. Provided context\n\n"
            "RULES:\n"
            "1. Answer ONLY using the provided CONTEXT blocks.\n"
            "2. NEVER use prior or external knowledge.\n"
            "3. If the answer is NOT explicitly present, respond EXACTLY with:\n"
            "   'Not found in the provided documents.'\n"
            "4. Do NOT guess, infer, or generalize.\n"
            "5. Prefer TABLE data over paragraph text for numeric or dosage answers.\n"
            "6. If any context has LOW or UNKNOWN confidence, clearly warn the user.\n"
            "7. Use simple, farmer-friendly language.\n"
            "8. Keep answers short, precise, and actionable.\n"
        )

    # ---------------- PROMPT BUILD ----------------

    def build(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        response_language: str = "same-as-question"
    ) -> Dict[str, str]:
        """
        Build the final prompt.
        Returns:
        - system_prompt
        - user_prompt
        - used_chunks
        """

        selected_chunks = self._select_context(retrieved_chunks)

        # ---------- REFUSAL PATH ----------
        if not selected_chunks:
            return {
                "system_prompt": self._system_rules(),
                "user_prompt": (
                    f"USER QUESTION:\n{query}\n\n"
                    "CONTEXT:\n"
                    "No relevant documents were retrieved.\n\n"
                    "INSTRUCTION:\n"
                    "Respond exactly with:\n"
                    "'Not found in the provided documents.'\n\n"
                    "ANSWER:"
                ),
                "used_chunks": []
            }

        context_block = self._format_context(selected_chunks)

        user_prompt = (
            f"USER QUESTION:\n{query}\n\n"
            f"CONTEXT:\n{context_block}\n\n"
            "DEVELOPER INSTRUCTIONS:\n"
            "- Answer strictly from the context above.\n"
            "- Cite sources using (Source, Page).\n"
            "- If information is unclear or low confidence, warn explicitly.\n"
            f"- Respond in language: {response_language}.\n\n"
            "ANSWER FORMAT:\n"
            "- Use bullet points if steps are involved.\n"
            "- Use a short paragraph otherwise.\n"
            "- End with a 'Sources' section.\n\n"
            "ANSWER:"
        )

        return {
            "system_prompt": self._system_rules(),
            "user_prompt": user_prompt,
            "used_chunks": selected_chunks
        }
