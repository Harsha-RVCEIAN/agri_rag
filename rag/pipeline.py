from typing import Dict, Optional

from rag.retriever import Retriever
from rag.prompt_builder import PromptBuilder
from rag.answer_generator import AnswerGenerator


# =========================================================
# GLOBAL THRESHOLDS
# =========================================================

FINAL_CONFIDENCE_THRESHOLD = 0.35

CATEGORY_CONFIDENCE_CAPS = {
    "policy": 1.0,
    "market": 0.9,
    "advisory": 0.65,
    "disease": 0.5,
}

MANDATORY_TEXT_DOMINANCE = {"policy", "market"}
OCR_FORBIDDEN_CATEGORIES = {"policy", "disease"}
TABLE_REQUIRED_CATEGORIES = {"market"}

FAST_CATEGORIES = {"policy"}  # definition / lookup ≠ full RAG


# =========================================================
# CATEGORY → DOMAIN MAP
# =========================================================

CATEGORY_DOMAIN_MAP = {
    "policy": "scheme",
    "market": "market",
    "advisory": "crop_production",
    "disease": "crop_disease",
}


# =========================================================
# USER-FACING FALLBACK MESSAGES
# =========================================================

NO_ANSWER_MESSAGES = {
    "retrieval_failed": {
        "message": "I couldn’t find relevant information in the documents for your question.",
        "suggestion": "Try rephrasing your question or using more specific terms."
    },
    "no_chunks": {
        "message": "The documents do not contain enough usable information to answer this.",
        "suggestion": "You may upload additional documents or ask a more focused question."
    },
    "hallucination_detected": {
        "message": "The available information is unclear or inconsistent, so I cannot answer safely.",
        "suggestion": "Please check the document quality or ask about a different section."
    },
    "low_final_confidence": {
        "message": "I found relevant information, but it is not reliable enough to answer confidently.",
        "suggestion": "Try narrowing your question or consult official sources."
    },
    "category_violation": {
        "message": "This question requires stricter evidence than what is available.",
        "suggestion": "Please consult an official source or provide authoritative documents."
    },
}


class RAGPipeline:
    """
    Final decision authority.
    """

    def __init__(self, vector_store, embedder):
        self.embedder = embedder
        self.retriever = Retriever(vector_store)
        self.prompt_builder = PromptBuilder()
        self.answer_generator = AnswerGenerator()

    # =====================================================
    # MAIN
    # =====================================================

    def run(
        self,
        query: str,
        intent: Optional[str] = None,
        language: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Dict:

        category = category or "policy"
        domain = CATEGORY_DOMAIN_MAP.get(category)
        fast_mode = category in FAST_CATEGORIES

        # -------------------------------------------------
        # RETRIEVAL
        # -------------------------------------------------
        query_vectors = self.embedder.embed_texts([query])

        retrieval = self.retriever.retrieve(
            query_vectors=query_vectors,
            intent=intent,
            language=language,
            domain=domain,
        )

        retrieval_diag = retrieval.get("diagnostics", {})
        diagnostics = {
            "category": category,
            "domain": domain,
            "fast_mode": fast_mode,
            "retrieval": retrieval_diag,
        }

        # ---------- HARD FAIL (NO DATA) ----------
        if retrieval_diag.get("status") != "ok":
            return self._fallback(
                "retrieval_failed",
                diagnostics,
                allow_fallback=True,
            )

        chunks = retrieval.get("chunks") or []
        if not chunks:
            return self._fallback(
                "no_chunks",
                diagnostics,
                allow_fallback=True,
            )

        retrieval_confidence = retrieval_diag.get("retrieval_confidence", 0.0)
        if retrieval_confidence <= 0.05:
            return self._fallback(
                "retrieval_failed",
                diagnostics,
                allow_fallback=True,
            )

        content_mix = retrieval_diag.get("content_mix", {})

        # -------------------------------------------------
        # CATEGORY GUARDS (DATA EXISTS → NO FALLBACK)
        # -------------------------------------------------
        if category in OCR_FORBIDDEN_CATEGORIES and content_mix.get("ocr", 0) > 0:
            return self._fallback(
                "category_violation",
                diagnostics,
                allow_fallback=False,
            )

        if category in TABLE_REQUIRED_CATEGORIES and content_mix.get("table_row", 0) == 0:
            return self._fallback(
                "category_violation",
                diagnostics,
                allow_fallback=False,
            )

        if category in MANDATORY_TEXT_DOMINANCE:
            if (content_mix.get("text", 0) + content_mix.get("procedure", 0)) == 0:
                return self._fallback(
                    "category_violation",
                    diagnostics,
                    allow_fallback=False,
                )

        # -------------------------------------------------
        # PROMPT
        # -------------------------------------------------
        prompt_bundle = self.prompt_builder.build(
            query=query,
            retrieved_chunks=chunks,
            fast_mode=fast_mode,
        )

        if not prompt_bundle or not prompt_bundle.get("used_chunks"):
            return self._fallback(
                "no_chunks",
                diagnostics,
                allow_fallback=True,
            )

        # -------------------------------------------------
        # ANSWER
        # -------------------------------------------------
        answer_result = self.answer_generator.generate(prompt_bundle)

        hallucinated = answer_result.get("hallucinated", False)
        answer_confidence = answer_result.get("confidence", 0.0)

        diagnostics["answer"] = {
            "answer_confidence": answer_confidence,
            "hallucinated": hallucinated,
            "used_chunks": len(answer_result.get("used_chunks", [])),
        }

        if hallucinated:
            return self._fallback(
                "hallucination_detected",
                diagnostics,
                allow_fallback=False,
            )

        # -------------------------------------------------
        # CONFIDENCE FUSION
        # -------------------------------------------------
        if fast_mode:
            final_confidence = round(
                0.65 * retrieval_confidence +
                0.35 * answer_confidence,
                3
            )
        else:
            final_confidence = round(
                0.55 * retrieval_confidence +
                0.45 * answer_confidence,
                3
            )

        final_confidence = min(
            final_confidence,
            CATEGORY_CONFIDENCE_CAPS.get(category, 1.0)
        )

        if final_confidence < FINAL_CONFIDENCE_THRESHOLD:
            return self._fallback(
                "low_final_confidence",
                diagnostics,
                confidence=final_confidence,
                allow_fallback=False,   # 🔑 DATA EXISTS
            )

        return {
            "status": "answer",
            "answer": answer_result.get("answer", ""),
            "confidence": final_confidence,
            "category": category,
            "domain": domain,
            "diagnostics": diagnostics,
        }

    # =====================================================
    # FALLBACK
    # =====================================================

    def _fallback(
        self,
        reason: str,
        diagnostics: Optional[Dict],
        confidence: float = 0.0,
        allow_fallback: bool = False,
    ) -> Dict:

        ux = NO_ANSWER_MESSAGES.get(
            reason,
            {
                "message": "I’m unable to answer this question with the available information.",
                "suggestion": "Please try rephrasing or provide additional documents."
            }
        )

        return {
            "status": "no_answer",
            "answer": None,
            "confidence": confidence,
            "message": ux["message"],
            "suggestion": ux["suggestion"],
            "reason": reason,

            # 🔑 CONTROL FLAG (API decides Gemini fallback)
            "allow_fallback": allow_fallback,

            "diagnostics": diagnostics or {},
        }