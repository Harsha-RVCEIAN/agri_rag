# scripts/run_pipeline.py

import os
import sys

# ---- ensure project root is on path ----
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from embeddings.embedder import Embedder
from embeddings.vector_store import VectorStore
from rag.pipeline import RAGPipeline


def main():
    print("🚀 Agri-RAG CLI Runner")
    print("Type 'exit' to quit.\n")

    # ---- init core components ----
    embedder = Embedder()
    vector_store = VectorStore()

    rag = RAGPipeline(
        vector_store=vector_store,
        embedder=embedder
    )

    while True:
        query = input("❓ Question: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        # ---- optional knobs ----
        intent = None        # e.g. "eligibility", "procedure"
        language = None      # e.g. "en"
        category = None      # e.g. "policy", "market", "disease"

        result = rag.run(
            query=query,
            intent=intent,
            language=language,
            category=category,
        )

        print("\n🧠 RESULT")
        print("────────────")

        if result["status"] == "answer":
            print("Answer:")
            print(result["answer"])
            print(f"\nConfidence: {result['confidence']}")
        else:
            print(result.get("message", "No answer."))
            print("Suggestion:", result.get("suggestion"))

        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
