from typing import Dict, Optional

from rag.retriever import Retriever
from rag.prompt_builder import PromptBuilder
from rag.answer_generator import AnswerGenerator


# ---------------- GLOBAL THRESHOLDS ----------------

FINAL_CONFIDENCE_THRESHOLD = 0.5

CATEGORY_CONFIDENCE_CAPS = {
    "policy": 1.0,
    "market": 0.9,
    "advisory": 0.65,
    "disease": 0.5,
}

MANDATORY_TEXT_DOMINANCE = {"policy", "market"}
OCR_FORBIDDEN_CATEGORIES = {"policy", "disease"}
TABLE_REQUIRED_CATEGORIES = {"market"}

FAST_CATEGORIES = {"policy"}  # definition is NOT RAG


# ---------------- USER-FACING FALLBACK MESSAGES ----------------

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
        "message": "I found some information, but it is not reliable enough to give a confident answer.",
        "suggestion": "Try narrowing your question or providing a more specific document."
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

    # ---------------- MAIN ----------------

    def run(
        self,
        query: str,
        intent: Optional[str] = None,
        language: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Dict:

        category = category or "policy"
        fast_mode = category in FAST_CATEGORIES

        # ---------------- RETRIEVAL ----------------
        # ✅ FIXED: always returns List[List[float]]
        query_vectors = self.embedder.embed_texts([query])  # List[List[float]]

        retrieval = self.retriever.retrieve(
            query=query,
            query_vectors=query_vectors,
            intent=intent,
            language=language
        )

        retrieval_diag = retrieval.get("diagnostics", {})
        diagnostics = {
            "category": category,
            "retrieval": retrieval_diag,
        }

        # ---------- HARD FAIL ----------
        if retrieval_diag.get("status") != "ok":
            return self._fallback("retrieval_failed", diagnostics)

        chunks = retrieval.get("chunks") or []
        if not chunks:
            return self._fallback("no_chunks", diagnostics)

        retrieval_confidence = retrieval_diag.get("retrieval_confidence", 0.0)
        if retrieval_confidence <= 0.05:
            return self._fallback("retrieval_failed", diagnostics)

        content_mix = retrieval_diag.get("content_mix", {})

        # ---------------- CATEGORY GUARDS ----------------

        if category in OCR_FORBIDDEN_CATEGORIES and content_mix.get("ocr", 0) > 0:
            return self._fallback("category_violation", diagnostics)

        if category in TABLE_REQUIRED_CATEGORIES and content_mix.get("table_row", 0) == 0:
            return self._fallback("category_violation", diagnostics)

        if category in MANDATORY_TEXT_DOMINANCE:
            if (content_mix.get("text", 0) + content_mix.get("procedure", 0)) == 0:
                return self._fallback("category_violation", diagnostics)

        # ---------------- PROMPT ----------------
        prompt_bundle = self.prompt_builder.build(
            query=query,
            retrieved_chunks=chunks
        )

        if not prompt_bundle or not prompt_bundle.get("used_chunks"):
            return self._fallback("no_chunks", diagnostics)

        # ---------------- ANSWER ----------------
        answer_result = self.answer_generator.generate(prompt_bundle)

        hallucinated = answer_result.get("hallucinated", False)
        answer_confidence = answer_result.get("confidence", 0.0)

        diagnostics["answer"] = {
            "answer_confidence": answer_confidence,
            "hallucinated": hallucinated,
            "used_chunks": len(answer_result.get("used_chunks", [])),
        }

        if hallucinated:
            return self._fallback("hallucination_detected", diagnostics)

        # ---------------- CONFIDENCE FUSION ----------------
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
                confidence=final_confidence
            )

        return {
            "status": "answer",
            "answer": answer_result.get("answer", ""),
            "confidence": final_confidence,
            "category": category,
            "diagnostics": diagnostics,
        }

    # ---------------- FALLBACK ----------------

    def _fallback(
        self,
        reason: str,
        diagnostics: Optional[Dict],
        confidence: float = 0.0,
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
            "diagnostics": diagnostics or {},
        }
