# scripts/run_pipeline.py

import sys
import os

# ---- ensure project root is on path ----
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rag.retriever import Retriever
from llm.answerer import generate_answer


def main():
    print("🚀 Agri-RAG Query Runner")

    # ---- user query ----
    query = "what is features of PMFBY?"

    # ---- retrieve evidence ----
    retriever = Retriever()
    result = retriever.retrieve(query)

    chunks = result.get("chunks", [])
    diagnostics = result.get("diagnostics", {})

    print(f"🔎 Retrieved chunks: {len(chunks)}")
    print(f"🧪 Diagnostics: {diagnostics}")

    if not chunks:
        print("\n❌ ANSWER:\nNot found in the provided documents.")
        return

    # ---- generate answer ----
    answer = generate_answer(query, chunks)
    print("\nquery:\n", query)
    print("\n🧠 ANSWER:\n")
    print(answer)


if __name__ == "__main__":
    main()
