from typing import Dict, List
import re
from llm.llm_client import LLMClient

# ---------------- CONFIG ----------------

MAX_ANSWER_CHARS = 800
MIN_ANSWER_TOKENS = 5

HALLUCINATION_MARKERS = [
    "i think",
    "i believe",
    "generally",
    "usually",
    "in most cases",
]

UNSAFE_CERTAINTY_MARKERS = [
    "you should",
    "must apply",
    "recommended dose",
    "always",
    "definitely",
    "exactly",
]

UNCERTAINTY_MARKERS = [
    "may",
    "might",
    "can vary",
    "possible",
    "depending on",
    "uncertain",
]

OCR_CONFIDENCE_PENALTY = 0.6


class AnswerGenerator:
    """
    Executes LLM and reports answer-level safety signals.
    Does NOT decide final acceptance.
    """

    def __init__(self):
        self.llm = LLMClient()

    # ---------------- MAIN ----------------

    def generate(self, prompt_bundle: Dict) -> Dict:
        answer = self.llm.generate(
            system_prompt=prompt_bundle["system_prompt"],
            user_prompt=prompt_bundle["user_prompt"],
            temperature=0.0,
            max_tokens=400
        )

        if not answer:
            return self._fail()

        answer = answer.strip()

        hallucinated = self._detect_hallucination(answer)

        confidence = self._compute_confidence(
            answer=answer,
            chunks=prompt_bundle.get("used_chunks", [])
        )

        # Unsafe certainty without uncertainty = hallucination
        if self._unsafe_certainty(answer) and not self._has_uncertainty(answer):
            hallucinated = True
            confidence = 0.0

        return {
            "answer": answer,
            "confidence": confidence,
            "hallucinated": hallucinated,
            "used_chunks": prompt_bundle.get("used_chunks", [])
        }

    # ---------------- CONFIDENCE ----------------

    def _compute_confidence(self, answer: str, chunks: List[Dict]) -> float:
        """
        Conservative confidence:
        - Evidence strength
        - Grounding
        - OCR penalty
        """

        if not chunks:
            return 0.0

        scores = [c.get("score", 0.0) for c in chunks]
        chunk_strength = sum(scores) / len(scores)

        grounding = self._grounding_ratio(answer, chunks)

        # OCR penalty
        ocr_ratio = sum(
            1 for c in chunks if c.get("content_type") == "ocr"
        ) / len(chunks)

        penalty = 1.0 - (ocr_ratio * OCR_CONFIDENCE_PENALTY)

        confidence = round(
            min(1.0, (0.6 * chunk_strength + 0.4 * grounding) * penalty),
            3
        )

        return confidence

    # ---------------- HALLUCINATION ----------------

    def _detect_hallucination(self, answer: str) -> bool:
        lowered = answer.lower()

        if len(answer) > MAX_ANSWER_CHARS:
            return True

        if len(answer.split()) < MIN_ANSWER_TOKENS:
            return True

        if any(m in lowered for m in HALLUCINATION_MARKERS):
            return True

        return False

    # ---------------- SAFETY CHECKS ----------------

    def _unsafe_certainty(self, answer: str) -> bool:
        lowered = answer.lower()
        return any(m in lowered for m in UNSAFE_CERTAINTY_MARKERS)

    def _has_uncertainty(self, answer: str) -> bool:
        lowered = answer.lower()
        return any(m in lowered for m in UNCERTAINTY_MARKERS)

    # ---------------- GROUNDING ----------------

    def _grounding_ratio(self, answer: str, chunks: List[Dict]) -> float:
        """
        Measures lexical grounding without trusting paraphrase.
        """

        answer_terms = set(re.findall(r"[a-zA-Z]{4,}", answer.lower()))
        if not answer_terms:
            return 0.0

        context_text = " ".join(
            c.get("text", "").lower() for c in chunks
        )

        supported = sum(1 for t in answer_terms if t in context_text)
        return supported / len(answer_terms)

    # ---------------- FAIL ----------------

    def _fail(self) -> Dict:
        return {
            "answer": "",
            "confidence": 0.0,
            "hallucinated": True,
            "used_chunks": []
        }
