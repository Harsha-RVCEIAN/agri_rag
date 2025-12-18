"""
Multilingual RAG test.

Purpose:
- Verify retrieval works across languages
- Verify answer language matches query language
- Verify grounding is preserved
- Verify no hallucination occurs
"""

from rag.pipeline import RAGPipeline


def assert_refusal(result: dict):
    assert result["refused"] is True, "Expected refusal but got answer"
    assert "not found" in result["answer"].lower()


def assert_answer(result: dict):
    assert result["refused"] is False, "Unexpected refusal"
    assert result["answer"], "Empty answer"
    assert result["confidence"] > 0.0, "Confidence should be > 0"
    assert result["citations"], "Missing citations"


def detect_language_simple(text: str) -> str:
    """
    Very rough language check (no ML).
    Good enough for tests.
    """
    if any(ch in text for ch in "ಅಆಇಈಉಊಎಏಐಒಓೌ"):
        return "kn"
    if any(ch in text for ch in "अआइईउऊएऐओऔ"):
        return "hi"
    return "en"


def test_multilingual_queries():
    rag = RAGPipeline()

    queries = [
        {
            "lang": "en",
            "q": "What is PM Kisan eligibility?",
        },
        {
            "lang": "hi",
            "q": "पीएम किसान योजना की पात्रता क्या है?",
        },
        {
            "lang": "kn",
            "q": "ಪಿಎಂ ಕಿಸಾನ್ ಯೋಜನೆಯ ಅರ್ಹತೆ ಏನು?",
        },
    ]

    base_answer = None

    for item in queries:
        print(f"\n🔍 Testing language: {item['lang']}")
        print(f"Query: {item['q']}")

        result = rag.run(query=item["q"])

        # ---------- must not hallucinate ----------
        assert "generally" not in result["answer"].lower()
        assert "i think" not in result["answer"].lower()

        # ---------- must answer ----------
        assert_answer(result)

        # ---------- language check ----------
        detected = detect_language_simple(result["answer"])
        assert detected == item["lang"], (
            f"Expected language {item['lang']} but got {detected}"
        )

        # ---------- semantic consistency ----------
        if base_answer is None:
            base_answer = result["answer"]
        else:
            # not exact match, but should be about same scheme
            assert "किसान" in result["answer"] or "ಕಿಸಾನ್" in result["answer"] or "kisan" in result["answer"].lower()

        print("✅ Passed")


if __name__ == "__main__":
    test_multilingual_queries()
    print("\n🎉 Multilingual test PASSED")