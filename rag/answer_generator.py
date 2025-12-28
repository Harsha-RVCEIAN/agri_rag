from typing import Dict, List
import re
from llm.llm_client import LLMClient

# ---------------- CONFIG ----------------

MAX_ANSWER_CHARS = 800
MIN_ANSWER_TOKENS = 5

HALLUCINATION_MARKERS = (
    "i think",
    "i believe",
    "generally",
    "usually",
    "in most cases",
)

UNSAFE_CERTAINTY_MARKERS = (
    "you should",
    "must apply",
    "recommended dose",
    "always",
    "definitely",
    "exactly",
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

# precompiled regex (latency-critical)
TERM_REGEX = re.compile(r"[a-zA-Z]{4,}")


class AnswerGenerator:
    """
    Executes LLM and reports answer-level safety signals.
    NO final decision logic here.
    """

    def __init__(self):
        self.llm = LLMClient()

    # ---------------- MAIN ----------------

    def generate(self, prompt_bundle: Dict) -> Dict:
        answer = self.llm.generate(
            system_prompt=prompt_bundle["system_prompt"],
            user_prompt=prompt_bundle["user_prompt"],
            temperature=0.0,
            max_tokens=256,  # 🔥 reduced safely
        )

        if not answer:
            return self._fail()

        answer = answer.strip()

        # ---------- FAST STRUCTURAL FAILS ----------
        if (
            len(answer) > MAX_ANSWER_CHARS or
            len(answer.split()) < MIN_ANSWER_TOKENS
        ):
            return self._fail()

        lowered = answer.lower()

        # ---------- HALLUCINATION LANGUAGE ----------
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

    # ---------------- CONFIDENCE ----------------

    def _compute_confidence(self, answer: str, chunks: List[Dict]) -> float:
        """
        Conservative confidence:
        - Evidence strength
        - Lexical grounding
        - OCR penalty
        """

        scores = [c.get("score", 0.0) for c in chunks]
        chunk_strength = sum(scores) / len(scores)

        # ---------- GROUNDING (EARLY EXIT) ----------
        grounding = self._grounding_ratio(answer, chunks)
        if grounding == 0.0:
            return 0.0

        # ---------- OCR PENALTY ----------
        ocr_ratio = sum(
            1 for c in chunks if c.get("content_type") == "ocr"
        ) / len(chunks)

        penalty = 1.0 - (ocr_ratio * OCR_CONFIDENCE_PENALTY)

        return round(
            min(1.0, (0.6 * chunk_strength + 0.4 * grounding) * penalty),
            3
        )

    # ---------------- SAFETY ----------------

    def _unsafe_certainty(self, lowered: str) -> bool:
        return any(m in lowered for m in UNSAFE_CERTAINTY_MARKERS)

    def _has_uncertainty(self, lowered: str) -> bool:
        return any(m in lowered for m in UNCERTAINTY_MARKERS)

    # ---------------- GROUNDING ----------------

    def _grounding_ratio(self, answer: str, chunks: List[Dict]) -> float:
        """
        Lexical grounding only.
        Fast and conservative.
        """

        answer_terms = set(TERM_REGEX.findall(answer.lower()))
        if not answer_terms:
            return 0.0

        context_text = " ".join(
            c.get("text", "").lower() for c in chunks
        )

        supported = 0
        for t in answer_terms:
            if t in context_text:
                supported += 1

        return supported / len(answer_terms)

    # ---------------- FAIL ----------------

    def _fail(self) -> Dict:
        return {
            "answer": "",
            "confidence": 0.0,
            "hallucinated": True,
            "used_chunks": []
        }
