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


# =========================================================
# CONFIG
# =========================================================

PDF_DIR = os.path.join(PROJECT_ROOT, "data", "pdfs")
STATE_FILE = os.path.join(PROJECT_ROOT, "data", "vector_store", "index_state.json")

SUPPORTED_EXTENSIONS = {".pdf"}


# =========================================================
# DOCUMENT-LEVEL INGESTION RULES  🔑
# =========================================================
# EVERY PDF MUST BE LISTED HERE
# NO DEFAULT FALLBACK
# =========================================================

DOCUMENT_RULES: Dict[str, Dict] = {

    # ---------- CROP PRODUCTION ----------
    "Rice.pdf": {
        "domain": "crop_production",
        "allow_tables": False,
        "confidence_cap": 1.0,
    },
    "WHEAT.pdf": {
        "domain": "crop_production",
        "allow_tables": False,
    },
    "Wheat2016.pdf": {
        "domain": "crop_production",
        "allow_tables": False,
    },
    "cotton.pdf": {
        "domain": "crop_production",
        "allow_tables": False,
    },

    # ---------- DISEASE ----------
    "crop disease.pdf": {
        "domain": "crop_disease",
        "allow_tables": False,
        "confidence_cap": 0.8,
    },
    "crop diseases.pdf": {
        "domain": "crop_disease",
        "allow_tables": False,
        "confidence_cap": 0.8,
    },

    # ---------- SCHEMES ----------
    "PMFBY_Guidelines.pdf": {
        "domain": "scheme",
        "allow_tables": False,
        "confidence_cap": 0.9,
    },
    "midh_Guidelines.pdf": {
        "domain": "scheme",
        "allow_tables": False,
    },

    # ---------- MARKET ----------
    "TomatoOctober2018.pdf": {
        "domain": "market",
        "summary_only": True,
        "allow_tables": False,
        "confidence_cap": 0.6,
    },

    # ---------- STATISTICS ----------
    "annual_report_english_2022_23.pdf": {
        "domain": "statistics",
        "summary_only": True,
        "confidence_cap": 0.5,
    },
}


# =========================================================
# STATE UTILS
# =========================================================

def compute_file_hash(path: str) -> str:
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


# =========================================================
# INGESTION
# =========================================================

def ingest_single_pdf(
    pdf_path: str,
    embedder: Embedder,
    vector_store: VectorStore,
    state: Dict[str, str]
) -> None:

    filename = os.path.basename(pdf_path)

    # 🔴 STRICT MODE — NO UNMAPPED FILES
    if filename not in DOCUMENT_RULES:
        raise ValueError(
            f"PDF '{filename}' is NOT defined in DOCUMENT_RULES. "
            f"Ingestion aborted to prevent domain leakage."
        )

    file_hash = compute_file_hash(pdf_path)

    if state.get(pdf_path) == file_hash:
        print(f"⏩ Skipping already indexed PDF: {filename}")
        return

    rules = DOCUMENT_RULES[filename]

    print(f"\n📄 Ingesting: {filename}")
    print(f"   → Rules: {rules}")

    # ---------- extract ----------
    chunks: List[Dict] = ingest_pdf(
        pdf_path,
        doc_rules=rules
    )

    if not chunks:
        raise RuntimeError(f"No usable chunks produced for {filename}")

    # ---------- enforce domain + confidence ----------
    cap = rules.get("confidence_cap", 1.0)
    domain = rules["domain"]

    for c in chunks:
        meta = c.get("metadata", {})
        meta["domain"] = domain
        meta["confidence"] = min(meta.get("confidence", 1.0), cap)

    print(f"   → Chunks produced: {len(chunks)}")

    # ---------- embed ----------
    embedded_chunks = embedder.embed_chunks(chunks)
    if not embedded_chunks:
        raise RuntimeError(f"Embedding failed for {filename}")

    # ---------- store ----------
    vector_store.upsert(embedded_chunks)
    print("   ✅ Stored in vector database")

    state[pdf_path] = file_hash


def ingest_all_pdfs() -> None:

    if not os.path.exists(PDF_DIR):
        raise FileNotFoundError(f"PDF directory not found: {PDF_DIR}")

    pdf_files = [
        f for f in os.listdir(PDF_DIR)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not pdf_files:
        print("⚠️ No PDFs found.")
        return

    embedder = Embedder()
    vector_store = VectorStore()
    state = load_state()

    print("\n🚜 Starting ingestion")
    print(f"📄 PDFs found: {len(pdf_files)}")

    for file in pdf_files:
        try:
            ingest_single_pdf(
                os.path.join(PDF_DIR, file),
                embedder,
                vector_store,
                state
            )
        except Exception:
            print(f"\n❌ Ingestion FAILED for {file}")
            traceback.print_exc()
            print("\n🛑 Stopping ingestion to prevent corruption.")
            break   # 🔑 STOP ON FIRST ERROR

    save_state(state)
    print("\n🎉 Ingestion finished safely.")


if __name__ == "__main__":
    ingest_all_pdfs()
