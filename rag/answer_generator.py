from typing import Dict, List
import re
from llm.llm_client import LLMClient

# ---------------- CONFIG ----------------

MAX_ANSWER_CHARS = 1200     # 🔑 increased (fixes cut answers)
MIN_ANSWER_TOKENS = 5

HALLUCINATION_MARKERS = (
    "you should",
    "must apply",
    "recommended dose",
    "always",
    "definitely",
    "exactly",
)

UNSAFE_CERTAINTY_MARKERS = (
    "you should",
    "must apply",
    "recommended dose",
    "always",
    "definitely",
    
)

UNCERTAINTY_MARKERS = (
    "may",
    "might",
    "can vary",
    "possible",
    "depending on",
    "uncertain",
)

OCR_CONFIDENCE_PENALTY = 0.6

TERM_REGEX = re.compile(r"[a-zA-Z]{4,}")


class AnswerGenerator:
    """
    Executes LLM and reports answer-level safety signals.
    NO routing or fallback decisions here.
    """

    def __init__(self):
        self.llm = LLMClient()              # local (RAG)
        self.fallback_llm = LLMClient(
            provider="gemini"              # 🔑 fallback AI
        )

    # =====================================================
    # MAIN (RAG ANSWER)
    # =====================================================

    def generate(self, prompt_bundle: Dict) -> Dict:
        answer = self.llm.generate(
            system_prompt=prompt_bundle["system_prompt"],
            user_prompt=prompt_bundle["user_prompt"],
            temperature=0.0,
            max_tokens=600,
        )

        if not answer:
            return self._fail()

        answer = answer.strip()

        # ---------- BASIC STRUCTURE GUARD ----------
        if len(answer.split()) < MIN_ANSWER_TOKENS:
            return self._fail()

        if len(answer) > MAX_ANSWER_CHARS:
            answer = answer[:MAX_ANSWER_CHARS].rsplit(".", 1)[0] + "."

        lowered = answer.lower()

        # ---------- SOFT HALLUCINATION CHECK ----------
        if any(m in lowered for m in HALLUCINATION_MARKERS):
            return self._fail()

        chunks = prompt_bundle.get("used_chunks", [])
        if not chunks:
            return self._fail()

        # ---------- UNSAFE CERTAINTY ----------
        if self._unsafe_certainty(lowered) and not self._has_uncertainty(lowered):
            return {
                "answer": answer,
                "confidence": 0.0,
                "hallucinated": True,
                "used_chunks": chunks,
            }

        # ---------- CONFIDENCE ----------
        confidence = self._compute_confidence(answer, chunks)

        if confidence == 0.0:
            return self._fail()

        return {
            "answer": answer,
            "confidence": confidence,
            "hallucinated": False,
            "used_chunks": chunks,
        }

    # =====================================================
    # GEMINI FALLBACK (SUMMARY MODE)
    # =====================================================

    def fallback_with_llm(self, query: str) -> str:
        """
        Used ONLY when RAG is weak or missing.
        Returns a SHORT, CLEAN, SUMMARIZED answer.
        """

        return self.fallback_llm.generate(
            system_prompt=(
                "You are an agricultural expert.\n"
                "RULES:\n"
                "1.Provide a SHORT and CLEAR answer.\n"
                "2.Summarize key points only.\n"
                "3.Avoid long explanations.\n"
                "4.Use bullet points if helpful."
                "5. Answer clearly and concisely.\n"
                "6. Provide a short and complete summary donot give half answers,  not long explanations.\n"
                "7. If listing points, include all key points briefly.\n"
                "8. Do NOT repeat ideas or add filler text.\n"
                "9. Stop when the answer is logically complete and where it ends at '.'.\n"
                "10. Never stop at mid-sentence or mid-thought.\n"
                "11. if it is extending token limit , then pick till its previous sentence and display it. \n"
                "12.Give a clear, factual definition.\n"
                "13.No advice. No recommendations."
            ),
            user_prompt=query,
            temperature=0.3,
            max_tokens=180,
        )

    # =====================================================
    # CONFIDENCE
    # =====================================================

    def _compute_confidence(self, answer: str, chunks: List[Dict]) -> float:
        scores = []

        for c in chunks:
            if "score" in c:
                scores.append(c["score"])
            else:
                meta = c.get("metadata", {})
                base = meta.get("confidence", 0.6)
                priority = meta.get("priority", 3)
                scores.append(base * (1 + (priority - 3) * 0.05))

        chunk_strength = sum(scores) / len(scores)

        grounding = self._grounding_ratio(answer, chunks)
        if grounding == 0.0:
            return 0.0

        ocr_ratio = sum(
            1 for c in chunks if c.get("content_type") == "ocr"
        ) / len(chunks)

        penalty = 1.0 - (ocr_ratio * OCR_CONFIDENCE_PENALTY)

        final = (0.6 * chunk_strength + 0.4 * grounding) * penalty
        return round(min(1.0, final), 3)

    # =====================================================
    # SAFETY
    # =====================================================

    def _unsafe_certainty(self, lowered: str) -> bool:
        return any(m in lowered for m in UNSAFE_CERTAINTY_MARKERS)

    def _has_uncertainty(self, lowered: str) -> bool:
        return any(m in lowered for m in UNCERTAINTY_MARKERS)

    # =====================================================
    # GROUNDING
    # =====================================================

    def _grounding_ratio(self, answer: str, chunks: List[Dict]) -> float:
        answer_terms = set(TERM_REGEX.findall(answer.lower()))
        if not answer_terms:
            return 0.0

        context_text = " ".join(
            c.get("text", "").lower() for c in chunks
        )

        supported = sum(1 for t in answer_terms if t in context_text)
        return supported / len(answer_terms)

    # =====================================================
    # FAIL
    # =====================================================

    def _fail(self) -> Dict:
        return {
            "answer": "",
            "confidence": 0.0,
            "hallucinated": True,
            "used_chunks": [],
        }
