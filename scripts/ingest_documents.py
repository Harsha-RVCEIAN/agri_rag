import os
import sys
import json
import hashlib
import traceback
from typing import List, Dict

# ---- ensure project root is on path ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from ingestion.pipeline import ingest_pdf
from embeddings.embedder import Embedder
from embeddings.vector_store import VectorStore


# ---------------- CONFIG ----------------

PDF_DIR = os.path.join(PROJECT_ROOT, "data", "pdfs")
STATE_FILE = os.path.join(PROJECT_ROOT, "data", "vector_store", "index_state.json")

SUPPORTED_EXTENSIONS = {".pdf"}


# ---------------- STATE UTILS ----------------

def compute_file_hash(path: str) -> str:
    """Stable hash to detect file changes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state() -> Dict[str, str]:
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE, "r") as f:
        return json.load(f)


def save_state(state: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ---------------- INGESTION ----------------

def ingest_single_pdf(
    pdf_path: str,
    embedder: Embedder,
    vector_store: VectorStore,
    state: Dict[str, str]
) -> None:
    file_hash = compute_file_hash(pdf_path)

    if state.get(pdf_path) == file_hash:
        print(f"⏩ Skipping already indexed PDF: {os.path.basename(pdf_path)}")
        return

    print(f"\n📄 Ingesting NEW / UPDATED PDF: {os.path.basename(pdf_path)}")

    # ---------- extract chunks ----------
    chunks: List[Dict] = ingest_pdf(pdf_path)
    print(f"   → Extracted {len(chunks)} chunks")

    if not chunks:
        print("   ⚠️  No usable chunks found. Skipping.")
        return

    # ---------- embed ----------
    embedded_chunks = embedder.embed_chunks(chunks)
    print(f"   → Embedded {len(embedded_chunks)} chunks")

    if not embedded_chunks:
        print("   ⚠️  No embeddings generated. Skipping.")
        return

    # ---------- store ----------
    vector_store.upsert(embedded_chunks)
    print("   ✅ Stored in vector database")

    # ---------- update state ----------
    state[pdf_path] = file_hash


def ingest_all_pdfs() -> None:
    if not os.path.exists(PDF_DIR):
        raise FileNotFoundError(
            f"PDF directory not found: {PDF_DIR}\n"
            "Create it and place PDFs inside."
        )

    pdf_files = [
        f for f in os.listdir(PDF_DIR)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not pdf_files:
        print("⚠️  No PDF files found in data/pdfs/")
        return

    embedder = Embedder()
    vector_store = VectorStore()
    state = load_state()

    print("\n🚜 Starting document ingestion")
    print(f"📂 PDF directory: {PDF_DIR}")
    print(f"📄 PDFs found: {len(pdf_files)}")

    for file in pdf_files:
        pdf_path = os.path.join(PDF_DIR, file)

        try:
            ingest_single_pdf(pdf_path, embedder, vector_store, state)

        except Exception:
            print(f"\n❌ Failed to ingest {file}")
            traceback.print_exc()

    save_state(state)
    print("\n🎉 Ingestion complete.")


# ---------------- ENTRY POINT ----------------

if __name__ == "__main__":
    ingest_all_pdfs()
