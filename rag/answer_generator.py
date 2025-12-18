from typing import Dict, List
from llm.llm_client import LLMClient


# ---------------- CONFIG ----------------

MIN_CONTEXT_CHUNKS = 1
LOW_CONFIDENCE_SCORE = 0.55

MAX_ANSWER_CHARS = 800

HALLUCINATION_MARKERS = [
    "i think",
    "i believe",
    "generally speaking",
    "in most cases",
    "usually",
]


# ---------------- ANSWER GENERATOR ----------------

class AnswerGenerator:
    """
    Responsible for:
    - Calling the LLM
    - Enforcing refusal logic
    - Attaching citations
    - Computing confidence
    - Blocking hallucinations
    """

    def __init__(self):
        self.llm = LLMClient()

    # ---------------- CONFIDENCE LOGIC ----------------

    def _compute_confidence(self, used_chunks: List[Dict]) -> float:
        """
        Conservative but agreement-aware confidence.
        """
        if not used_chunks:
            return 0.0

        scores = [c.get("score", 0.0) for c in used_chunks]
        top_score = max(scores)
        avg_score = sum(scores) / len(scores)

        # OCR penalty
        ocr_ratio = sum(
            1 for c in used_chunks if c.get("content_type") == "ocr"
        ) / len(used_chunks)

        # Agreement bonus: multiple strong chunks agreeing
        agreement_bonus = 0.05 * (len(used_chunks) - 1)

        confidence = (
            0.6 * top_score +
            0.3 * avg_score +
            agreement_bonus
        ) * (1.0 - 0.4 * ocr_ratio)

        return round(max(0.0, min(confidence, 1.0)), 2)

    # ---------------- REFUSAL CHECK ----------------

    def _should_refuse(self, used_chunks: List[Dict]) -> bool:
        if not used_chunks:
            return True

        if len(used_chunks) < MIN_CONTEXT_CHUNKS:
            return True

        # allow recovery if multiple chunks compensate
        top = used_chunks[0].get("score", 0.0)
        avg = sum(c["score"] for c in used_chunks) / len(used_chunks)

        if top < LOW_CONFIDENCE_SCORE and avg < LOW_CONFIDENCE_SCORE:
            return True

        return False

    # ---------------- LLM CALL ----------------

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        return self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,   # DO NOT TOUCH
            max_tokens=400
        )

    # ---------------- HALLUCINATION GUARD ----------------

    def _contains_hallucination(self, text: str) -> bool:
        lowered = text.lower()

        # block speculative language
        if any(marker in lowered for marker in HALLUCINATION_MARKERS):
            return True

        # block unsupported prescriptions
        if "should" in lowered or "recommended" in lowered:
            if "source" not in lowered:
                return True

        return False

    # ---------------- MAIN ENTRY ----------------

    def generate(self, prompt_bundle: Dict) -> Dict:
        system_prompt = prompt_bundle["system_prompt"]
        user_prompt = prompt_bundle["user_prompt"]
        used_chunks = prompt_bundle["used_chunks"]

        # ---------- pre-LLM refusal ----------
        if self._should_refuse(used_chunks):
            return {
                "answer": "Not found in the provided documents.",
                "confidence": 0.0,
                "citations": [],
                "used_chunks": [],
                "refused": True,
                "refusal_reason": "weak_or_missing_context"
            }

        # ---------- LLM call ----------
        answer = self._call_llm(system_prompt, user_prompt)

        if not answer:
            return {
                "answer": "Not found in the provided documents.",
                "confidence": 0.0,
                "citations": [],
                "used_chunks": [],
                "refused": True,
                "refusal_reason": "llm_failure"
            }

        answer = answer.strip()

        # ---------- post-LLM guards ----------
        if (
            "not found in the provided documents" in answer.lower()
            or len(answer) > MAX_ANSWER_CHARS
            or self._contains_hallucination(answer)
        ):
            return {
                "answer": "Not found in the provided documents.",
                "confidence": 0.0,
                "citations": [],
                "used_chunks": [],
                "refused": True,
                "refusal_reason": "hallucination_or_drift"
            }

        # ---------- confidence ----------
        confidence = self._compute_confidence(used_chunks)

        # ---------- citations ----------
        citations = [
            {
                "chunk_id": c["chunk_id"],
                "source": c.get("source"),
                "page": c.get("page"),
                "content_type": c.get("content_type"),
            }
            for c in used_chunks
        ]

        return {
            "answer": answer,
            "confidence": confidence,
            "citations": citations,
            "used_chunks": used_chunks,
            "refused": False
        }
