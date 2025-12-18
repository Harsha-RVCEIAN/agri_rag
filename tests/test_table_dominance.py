"""
Table Dominance Test

Purpose:
- Ensure TABLE data dominates over paragraph text for numeric questions
- Prevent paragraph hallucination when tables are available
- Verify retriever + reranker bias is working correctly
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ingestion.pipeline import ingest_pdf
from rag.pipeline import RAGPipeline

def test_ingest_produces_chunks():
    chunks = ingest_pdf("data/pdfs/rice_cultivation.pdf")
    print("CHUNKS:", chunks)
    assert len(chunks) > 0, "Ingestion returned ZERO chunks"

NUMERIC_TABLE_QUESTIONS = [
    "What is the urea dosage for rice at tillering stage?",
    "How much DAP fertilizer is required per acre for paddy?",
    "What is the seed rate for rice per acre?",
]


def test_table_dominance():
    rag = RAGPipeline()

    for q in NUMERIC_TABLE_QUESTIONS:
        print(f"\n🧪 Testing table dominance:")
        print(f"Question: {q}")

        result = rag.run(query=q)

        # ---------- must not hallucinate ----------
        answer = result.get("answer", "").lower()
        assert "i think" not in answer
        assert "generally" not in answer

        # ---------- must not silently guess ----------
        if result.get("refused"):
            print("✅ Refused due to weak or missing table data (SAFE)")
            continue

        # ---------- must answer ----------
        assert result.get("confidence", 0.0) > 0.0, "Answered but confidence is zero"
        assert result.get("citations"), "Answered but no citations"

        # ---------- TABLE MUST DOMINATE ----------
        citations = result["citations"]

        table_sources = [
            c for c in citations
            if c.get("content_type") in {"table_row", "table"}
        ]

        assert table_sources, (
            "Numeric answer given without table-based citations"
        )

        # ---------- paragraphs must NOT dominate ----------
        paragraph_only = all(
            c.get("content_type") == "text"
            for c in citations
        )

        assert not paragraph_only, (
            "Paragraph text dominated numeric answer instead of tables"
        )

        # ---------- no prescriptive language ----------
        forbidden_phrases = [
            "you should apply",
            "recommended dose",
            "always apply",
            "must apply",
        ]

        for phrase in forbidden_phrases:
            assert phrase not in answer, (
                f"Unsafe prescriptive phrase detected: '{phrase}'"
            )

        print("✅ Passed (table dominance enforced)")


if __name__ == "__main__":
    test_table_dominance()
    print("\n🎉 Table dominance test PASSED")