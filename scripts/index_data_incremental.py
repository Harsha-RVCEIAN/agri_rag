import os
from ingestion.pipeline import ingest_pdf
from embeddings.embedder import Embedder
from embeddings.vector_store import VectorStore
from utils.index_state import (
    load_state,
    save_state,
    is_new_or_modified,
    update_state
)

PDF_DIR = "data/pdfs"

def main():
    state = load_state()
    embedder = Embedder()
    vector_store = VectorStore()

    total_new_chunks = 0

    for file in os.listdir(PDF_DIR):
        if not file.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(PDF_DIR, file)

        if not is_new_or_modified(pdf_path, state):
            print(f"⏩ Skipping unchanged PDF: {file}")
            continue

        print(f"📄 Processing NEW/MODIFIED PDF: {file}")

        chunks = ingest_pdf(pdf_path)
        records = embedder.embed_chunks(chunks)

        if records:
            vector_store.upsert(records)
            update_state(pdf_path, state)
            total_new_chunks += len(records)

    save_state(state)
    print(f"✅ Indexing complete. New vectors added: {total_new_chunks}")

if __name__ == "__main__":
    main()
