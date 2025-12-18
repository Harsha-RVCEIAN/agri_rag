import os
import sys
import shutil
import hashlib
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, List

# ---- ensure project root ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag.retriever import Retriever
from llm.answerer import generate_answer
from ingestion.pipeline import ingest_pdf
from embeddings.embedder import Embedder
from embeddings.vector_store import VectorStore


# ---------------- APP INIT ----------------

app = FastAPI(
    title="Agri-RAG API",
    description="Production-ready Agricultural RAG API",
    version="1.0.0"
)

# ---- load heavy components ONCE ----
retriever = Retriever()
embedder = Embedder()
vector_store = VectorStore()

PDF_DIR = os.path.join(PROJECT_ROOT, "data", "pdfs")
STATE_FILE = os.path.join(PROJECT_ROOT, "data", "vector_store", "index_state.json")

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)


# ---------------- STATE UTILS ----------------

def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_state() -> Dict[str, str]:
    if not os.path.exists(STATE_FILE):
        return {}
    import json
    with open(STATE_FILE, "r") as f:
        return json.load(f)


def _save_state(state: Dict[str, str]):
    import json
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ---------------- SCHEMAS ----------------

class ChatRequest(BaseModel):
    question: str
    intent: Optional[str] = None
    language: Optional[str] = None
    top_k: Optional[int] = 6


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    citations: List[Dict]
    refused: bool
    refusal_reason: Optional[str] = None
    diagnostics: Optional[Dict] = None


# ---------------- INGESTION (BACKGROUND) ----------------

def ingest_pdf_background(pdf_path: str):
    state = _load_state()
    file_hash = _file_hash(pdf_path)

    if state.get(pdf_path) == file_hash:
        return  # already indexed

    chunks = ingest_pdf(pdf_path)
    if not chunks:
        return

    records = embedder.embed_chunks(chunks)
    if not records:
        return

    vector_store.upsert(records)
    state[pdf_path] = file_hash
    _save_state(state)


# ---------------- ROUTES ----------------

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Agri-RAG API running"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = retriever.retrieve(
        query=req.question,
        intent=req.intent,
        language=req.language,
        top_k=req.top_k
    )

    chunks = result.get("chunks", [])
    diagnostics = result.get("diagnostics", {})

    if not chunks:
        return ChatResponse(
            answer="Not found in the provided documents.",
            confidence=0.0,
            citations=[],
            refused=True,
            refusal_reason="No relevant documents found",
            diagnostics=diagnostics
        )

    answer = generate_answer(req.question, chunks)

    # ---- confidence heuristic ----
    confidence = min(1.0, max(c["score"] for c in chunks))

    citations = [
        {
            "source": c.get("source"),
            "page": c.get("page"),
            "score": c.get("score")
        }
        for c in chunks
    ]

    return ChatResponse(
        answer=answer,
        confidence=confidence,
        citations=citations,
        refused=False,
        diagnostics=diagnostics
    )


@app.post("/upload-pdf")
def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    pdf_path = os.path.join(PDF_DIR, file.filename)

    # ---- save file ----
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ---- schedule background ingestion ----
    background_tasks.add_task(ingest_pdf_background, pdf_path)

    return {
        "status": "accepted",
        "file": file.filename,
        "message": "PDF uploaded. Ingestion running in background."
    }
