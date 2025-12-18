"""
No-Answer (Refusal) Test

Purpose:
- Verify the system REFUSES when information is not present
- Ensure no hallucination
- Ensure refusal is explicit, clean, and safe
"""

from rag.pipeline import RAGPipeline


def test_no_answer():
    rag = RAGPipeline()

    # Questions that MUST NOT exist in your documents
    impossible_questions = [
        "What is the quantum efficiency of photosynthesis on Mars?",
        "Explain blockchain consensus algorithms in agriculture subsidies",
        "What is the GPU requirement for running PM Kisan scheme?",
        "How does CRISPR gene editing work in rice farming?",
    ]

    for q in impossible_questions:
        print(f"\n🧪 Testing no-answer case:")
        print(f"Question: {q}")

        result = rag.run(query=q)

        # ---------- MUST refuse ----------
        assert result["refused"] is True, "System should refuse but did not"

        # ---------- exact refusal phrase ----------
        answer = result.get("answer", "").lower()
        assert "not found in the provided documents" in answer, (
            "Refusal message is incorrect or missing"
        )

        # ---------- no confidence ----------
        assert result.get("confidence", 1.0) == 0.0, (
            "Confidence must be 0.0 for refused answers"
        )

        # ---------- no citations ----------
        assert not result.get("citations"), (
            "Citations should be empty for refused answers"
        )

        # ---------- refusal reason ----------
        assert result.get("refusal_reason"), (
            "Refusal reason must be provided"
        )

        # ---------- hallucination guard ----------
        forbidden_phrases = [
            "generally",
            "usually",
            "i think",
            "it is recommended",
            "you should",
        ]

        for phrase in forbidden_phrases:
            assert phrase not in answer, (
                f"Hallucination phrase detected: '{phrase}'"
            )

        print("✅ Passed")


if __name__ == "__main__":
    test_no_answer()
    print("\n🎉 No-answer test PASSED")