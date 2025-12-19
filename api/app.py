import os
import sys
import shutil
import hashlib
from typing import Optional, Dict, List

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---- ensure project root ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag.retriever import Retriever
from llm.answerer import generate_answer
from llm.llm_client import LLMClient
from ingestion.pipeline import ingest_pdf
from embeddings.embedder import Embedder
from embeddings.vector_store import VectorStore


# ---------------- APP INIT ----------------

app = FastAPI(
    title="Agri-RAG API",
    description="Production-ready Agricultural RAG API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PDF_DIR = os.path.join(PROJECT_ROOT, "data", "pdfs")
STATE_FILE = os.path.join(PROJECT_ROOT, "data", "vector_store", "index_state.json")

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)


# ---------------- STARTUP ----------------

@app.on_event("startup")
def startup_event():
    try:
        app.state.embedder = Embedder()
        app.state.vector_store = VectorStore()
        app.state.retriever = Retriever(vector_store=app.state.vector_store)

        # LLMs
        app.state.local_llm = LLMClient(provider="local")
        app.state.gemini_llm = LLMClient(provider="gemini")

        print("✅ Application startup complete")

    except Exception as e:
        print("❌ Startup failed:", e)
        raise


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


# ---------------- ROUTES ----------------

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Agri-RAG API running"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # ---------- EMBED QUERY ----------
    embedder = app.state.embedder
    query_vectors = embedder.embed_texts([req.question])

    # ---------- RETRIEVE ----------
    result = app.state.retriever.retrieve(
        query=req.question,
        query_vectors=query_vectors,
        intent=req.intent,
        language=req.language,
        top_k=req.top_k
    )

    chunks = result.get("chunks", [])
    diagnostics = result.get("diagnostics", {})

    # ---------- RAG ANSWER ----------
    rag_result = generate_answer(
        query=req.question,
        docs=chunks,
        retrieval_diagnostics=diagnostics
    )

    # ---------- FALLBACK DECISION ----------
    if rag_result["allow_fallback"]:
        fallback_answer = app.state.gemini_llm.generate(
            system_prompt="You are an agricultural assistant.",
            user_prompt=req.question,
            temperature=0.3,
            max_tokens=300
        )

        return ChatResponse(
            answer=(
                "⚠️ Source: Generated using a general AI model.\n\n"
                + fallback_answer
            ),
            confidence=0.0,
            citations=[],
            refused=False,
            diagnostics={"fallback": "gemini"}
        )

    # ---------- NORMAL RAG RESPONSE ----------
    confidence = max((c["score"] for c in chunks), default=0.0)

    citations = [
        {
            "source": c.get("source"),
            "page": c.get("page"),
            "score": c.get("score")
        }
        for c in chunks
    ]

    return ChatResponse(
        answer=rag_result["answer"],
        confidence=min(1.0, confidence),
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

    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(ingest_pdf_background, pdf_path)

    return {
        "status": "accepted",
        "file": file.filename,
        "message": "PDF uploaded. Ingestion running in background."
    }


# ---------------- BACKGROUND TASK ----------------

def ingest_pdf_background(pdf_path: str):
    embedder = app.state.embedder
    vector_store = app.state.vector_store

    state = _load_state()
    file_hash = _file_hash(pdf_path)

    if state.get(pdf_path) == file_hash:
        return

    chunks = ingest_pdf(pdf_path)
    if not chunks:
        return

    records = embedder.embed_chunks(chunks)
    if not records:
        return

    vector_store.upsert(records)
    state[pdf_path] = file_hash
    _save_state(state)
