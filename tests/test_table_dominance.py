"""
Table Dominance Adversarial Test

Evaluator intent:
- Tables must dominate numeric answers
- Conflicting table/text must REFUSE
"""

from rag.pipeline import RAGPipeline


NUMERIC_QUESTIONS = [
    "What is the urea dosage for rice at tillering stage?",
    "What is the seed rate for rice per acre?",
]


def test_table_dominance_adversarial():
    rag = RAGPipeline()

    for q in NUMERIC_QUESTIONS:
        result = rag.run(query=q)

        if result["status"] == "no_answer":
            # refusal is SAFE and acceptable
            continue

        diagnostics = result.get("diagnostics", {})
        mix = diagnostics.get("retrieval", {}).get("content_mix", {})

        # tables must be present
        assert mix.get("table_row", 0) > 0, (
            "Numeric answer without table evidence"
        )

        # no prescriptive language
        forbidden = ["you should", "recommended", "always apply"]
        for f in forbidden:
            assert f not in result["answer"].lower()
