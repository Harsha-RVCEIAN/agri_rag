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
    "definition": 0.7,
}

MANDATORY_TEXT_DOMINANCE = {"policy", "market"}
OCR_FORBIDDEN_CATEGORIES = {"policy", "disease"}
TABLE_REQUIRED_CATEGORIES = {"market"}

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
    Enforces category-specific correctness rules.
    """

    def __init__(self):
        self.retriever = Retriever()
        self.prompt_builder = PromptBuilder()
        self.answer_generator = AnswerGenerator()

    # ---------------- MAIN ----------------

    def run(
        self,
        query: str,
        intent: Optional[str] = None,
        language: Optional[str] = None,
        category: Optional[str] = None,  # 🔑 policy | market | advisory | disease | definition
    ) -> Dict:

        category = category or "policy"  # conservative default

        # ---------------- RETRIEVAL ----------------
        retrieval = self.retriever.retrieve(
            query=query,
            intent=intent,
            language=language
        )

        retrieval_diag = retrieval.get("diagnostics", {})
        diagnostics = {"retrieval": retrieval_diag, "category": category}

        if retrieval_diag.get("status") == "fail":
            return self._fallback("retrieval_failed", diagnostics=diagnostics)

        chunks = retrieval.get("chunks", [])
        if not chunks:
            return self._fallback("no_chunks", diagnostics=diagnostics)

        retrieval_confidence = retrieval_diag.get("retrieval_confidence", 0.0)
        content_mix = retrieval_diag.get("content_mix", {})

        # ---------------- CATEGORY GUARDS ----------------

        # OCR forbidden
        if category in OCR_FORBIDDEN_CATEGORIES and content_mix.get("ocr", 0) > 0:
            return self._fallback("category_violation", diagnostics=diagnostics)

        # Table required
        if category in TABLE_REQUIRED_CATEGORIES and content_mix.get("table_row", 0) == 0:
            return self._fallback("category_violation", diagnostics=diagnostics)

        # Text dominance required
        if category in MANDATORY_TEXT_DOMINANCE:
            if content_mix.get("text", 0) + content_mix.get("procedure", 0) == 0:
                return self._fallback("category_violation", diagnostics=diagnostics)

        # ---------------- PROMPT ----------------
        prompt_bundle = self.prompt_builder.build(
            query=query,
            retrieved_chunks=chunks
        )

        # ---------------- ANSWER ----------------
        answer_result = self.answer_generator.generate(prompt_bundle)

        answer_confidence = answer_result.get("confidence", 0.0)
        hallucinated = answer_result.get("hallucinated", False)

        diagnostics["answer"] = {
            "answer_confidence": answer_confidence,
            "hallucinated": hallucinated,
            "used_chunks": len(answer_result.get("used_chunks", [])),
        }

        if hallucinated:
            return self._fallback("hallucination_detected", diagnostics=diagnostics)

        # ---------------- CONFIDENCE FUSION ----------------
        final_confidence = round(
            0.55 * retrieval_confidence +
            0.45 * answer_confidence,
            3
        )

        # ---------------- CATEGORY CONFIDENCE CAP ----------------
        cap = CATEGORY_CONFIDENCE_CAPS.get(category, 1.0)
        final_confidence = min(final_confidence, cap)

        if final_confidence < FINAL_CONFIDENCE_THRESHOLD:
            return self._fallback(
                "low_final_confidence",
                confidence=final_confidence,
                diagnostics=diagnostics
            )

        return {
            "status": "answer",
            "answer": answer_result.get("answer", ""),
            "confidence": final_confidence,
            "category": category,
            "diagnostics": diagnostics
        }

    # ---------------- FALLBACK ----------------

    def _fallback(
        self,
        reason: str,
        confidence: float = 0.0,
        diagnostics: Optional[Dict] = None
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
            "diagnostics": diagnostics or {}
        }
