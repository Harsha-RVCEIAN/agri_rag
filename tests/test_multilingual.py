"""
Multilingual Adversarial Test

Evaluator intent:
- Language of answer MUST match query
- System must refuse if grounding is weak
- Translation guessing = FAIL
"""

from rag.pipeline import RAGPipeline


def detect_language_simple(text: str) -> str:
    if any(ch in text for ch in "ಅಆಇಈಉಊಎಏಐಒಓೌ"):
        return "kn"
    if any(ch in text for ch in "अआइईउऊएऐओऔ"):
        return "hi"
    return "en"


def test_multilingual_adversarial():
    rag = RAGPipeline()

    queries = [
        ("en", "What is PM Kisan eligibility?"),
        ("hi", "पीएम किसान योजना की पात्रता क्या है?"),
        ("kn", "ಪಿಎಂ ಕಿಸಾನ್ ಯೋಜನೆಯ ಅರ್ಹತೆ ಏನು?"),
    ]

    for lang, q in queries:
        result = rag.run(query=q)

        if result["status"] == "no_answer":
            # SAFE refusal is acceptable
            assert result["confidence"] == 0.0
            continue

        answer = result["answer"]
        detected = detect_language_simple(answer)

        assert detected == lang, (
            f"Language mismatch: expected {lang}, got {detected}"
        )

        # hallucination guard
        forbidden = ["i think", "generally", "usually"]
        for f in forbidden:
            assert f not in answer.lower()
