"""
Adversarial No-Answer Test

Evaluator intent:
- Impossible questions MUST refuse
- Refusal must be explicit + confidence = 0
"""

from rag.pipeline import RAGPipeline


IMPOSSIBLE_QUESTIONS = [
    "What is the quantum efficiency of photosynthesis on Mars?",
    "Explain blockchain consensus algorithms in agriculture subsidies",
    "What GPU is required to run PM Kisan scheme?",
    "How does CRISPR gene editing work in rice farming?",
]


def test_no_answer_adversarial():
    rag = RAGPipeline()

    for q in IMPOSSIBLE_QUESTIONS:
        result = rag.run(query=q)

        assert result["status"] == "no_answer"
        assert result["confidence"] == 0.0

        # must guide user
        assert result.get("message")
        assert result.get("suggestion")

        # must not hallucinate
        forbidden = ["generally", "usually", "i think", "you should"]
        msg = (result.get("message") or "").lower()
        for f in forbidden:
            assert f not in msg
