# scripts/run_pipeline.py

import sys
import os
from typing import List, Dict

# ---- fix import path ----
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ingestion.pipeline import ingest_pdf
from embeddings.embedder import Embedder
from rag.retriever import Retriever
from llm.answerer import generate_answer

PDF_DIR = "data/pdfs"


def ingest_all_pdfs(pdf_dir: str) -> List[Dict]:
    all_chunks: List[Dict] = []

    for file in os.listdir(pdf_dir):
        if not file.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(pdf_dir, file)
        print(f"📄 Ingesting: {pdf_path}")

        chunks = ingest_pdf(pdf_path)
        all_chunks.extend(chunks)

    return all_chunks


def main():
    print("🚀 Starting Agri-RAG Pipeline")

    # 1️⃣ Ingest all PDFs
    chunks = ingest_all_pdfs(PDF_DIR)
    print(f"✅ Total chunks: {len(chunks)}")

    if not chunks:
        print("❌ No chunks ingested. Exiting.")
        return

    # 2️⃣ Embed chunks
    embedder = Embedder()
    records = embedder.embed_chunks(chunks)

    print(f"✅ Embedded vectors: {len(records)}")

    if not records:
        print("❌ No embeddings generated. Exiting.")
        return

    # ⚠️ At this point you SHOULD upsert to vector DB
    # vector_store.upsert(records)

    # 3️⃣ Query
    query = "What is nitrogen deficiency in rice?"

    # 4️⃣ Retrieve relevant chunks
    retriever = Retriever()
    result = retriever.retrieve(query)

    docs = result.get("chunks", [])
    diagnostics = result.get("diagnostics", {})

    print(f"🔎 Retrieved chunks: {len(docs)}")
    print(f"🧪 Retrieval diagnostics: {diagnostics}")

    if not docs:
        print("❌ No relevant chunks retrieved. Exiting.")
        return

    # 5️⃣ Generate answer
    answer = generate_answer(query, docs)

    print("\n🧠 ANSWER:\n")
    print(answer)


if __name__ == "__main__":
    main()
