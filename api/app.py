import os
import sys
import shutil
import hashlib
import re
from typing import Optional, Dict
from functools import lru_cache

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------- PATH SETUP ----------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------- INTERNAL IMPORTS ----------------

from rag.pipeline import RAGPipeline
from llm.llm_client import LLMClient
from ingestion.pipeline import ingest_pdf
from embeddings.embedder import Embedder
from embeddings.vector_store import VectorStore

# ---------------- APP INIT ----------------

app = FastAPI(
    title="Agri-RAG API",
    description="Evidence-bound agricultural QA system",
    version="2.2.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- STORAGE ----------------

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
STATE_FILE = os.path.join(DATA_DIR, "vector_store", "index_state.json")

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

# ---------------- STATE UTILS (🔥 FIXED) ----------------

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


def _save_state(state: Dict[str, str]) -> None:
    import json
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

# ---------------- STARTUP ----------------

@app.on_event("startup")
def startup_event():
    app.state.embedder = Embedder()
    app.state.vector_store = VectorStore()
    app.state.rag = RAGPipeline()
    app.state.gemini_llm = LLMClient(provider="gemini")
    print("✅ Agri-RAG API ready")

# ---------------- DOMAIN GUARDS ----------------

AGRI_KEYWORDS = {
    "agriculture", "farming", "crop", "soil", "fertilizer",
    "rice", "wheat", "maize", "paddy", "tomato", "potato",
    "organic", "irrigation", "yield", "harvest",
    "scheme", "pm kisan", "pmfby", "insurance",
}

PERSON_PATTERNS = (
    r"\bwho is\b", r"\bbiography\b", r"\bborn\b",
    r"\bage\b", r"\bnet worth\b"
)

DEFINITION_TRIGGERS = (
    "what is", "define", "definition of", "meaning of", "explain"
)

def is_agriculture_query(q: str) -> bool:
    q = q.lower()
    return any(k in q for k in AGRI_KEYWORDS)

def looks_like_person_query(q: str) -> bool:
    q = q.lower()
    return any(re.search(p, q) for p in PERSON_PATTERNS)

def is_definition_query(q: str) -> bool:
    return q.lower().strip().startswith(DEFINITION_TRIGGERS)

# ---------------- DEFINITION CACHE ----------------

@lru_cache(maxsize=256)
def cached_definition(question: str) -> str:
    return app.state.gemini_llm.generate(
        system_prompt=(
            "You are an agricultural expert.\n"
            "Give a clear, textbook-style definition.\n"
            "No advice. No recommendations. No sources."
        ),
        user_prompt=question,
        temperature=0.3,
        max_tokens=220,
    ).strip()

# ---------------- RAG CACHE ----------------
# ⚠️ NOTE: cache is cleared automatically on restart.
# Good enough for demo + evaluation.

@lru_cache(maxsize=512)
def cached_rag_answer(
    question: str,
    intent: Optional[str],
    language: Optional[str],
    category: Optional[str],
) -> Dict:
    return app.state.rag.run(
        query=question,
        intent=intent,
        language=language,
        category=category,
    )

# ---------------- SCHEMAS ----------------

class ChatRequest(BaseModel):
    question: str
    intent: Optional[str] = None
    language: Optional[str] = None
    category: Optional[str] = None

class ChatResponse(BaseModel):
    status: str
    answer: Optional[str]
    confidence: float
    message: Optional[str] = None
    suggestion: Optional[str] = None
    diagnostics: Optional[Dict] = None

# ---------------- ROUTES ----------------

@app.get("/")
def health():
    return {"status": "ok"}

# ---------------- CHAT ----------------

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    raw = req.question.strip()
    if not raw:
        raise HTTPException(400, "Empty question")

    # ---------- FAST HARD REJECT ----------
    if looks_like_person_query(raw):
        return ChatResponse(
            status="no_answer",
            answer=None,
            confidence=0.0,
            message="This system answers agriculture-related questions only.",
            suggestion="Ask about crops, farming practices, or schemes."
        )

    if not is_agriculture_query(raw):
        return ChatResponse(
            status="no_answer",
            answer=None,
            confidence=0.0,
            message="This system is limited to agriculture topics.",
            suggestion="Rephrase with an agriculture focus."
        )

    # ---------- FAST DEFINITION ----------
    if is_definition_query(raw):
        return ChatResponse(
            status="answer",
            answer=cached_definition(raw),
            confidence=0.7,
            diagnostics={"category": "definition"}
        )

    # ---------- CACHED RAG ----------
    result = cached_rag_answer(
        raw,
        req.intent,
        req.language,
        req.category
    )

    if result["status"] == "answer":
        return ChatResponse(
            status="answer",
            answer=result["answer"],
            confidence=result["confidence"],
            diagnostics=result.get("diagnostics")
        )

    return ChatResponse(
        status="no_answer",
        answer=None,
        confidence=result.get("confidence", 0.0),
        message=result.get("message"),
        suggestion=result.get("suggestion"),
        diagnostics=result.get("diagnostics")
    )

# ---------------- PDF UPLOAD ----------------

@app.post("/upload-pdf")
def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files allowed")

    path = os.path.join(PDF_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    background_tasks.add_task(ingest_pdf_background, path)
    return {"status": "accepted", "file": file.filename}

# ---------------- INGEST ----------------

def ingest_pdf_background(pdf_path: str):
    state = _load_state()
    h = _file_hash(pdf_path)

    if state.get(pdf_path) == h:
        return

    chunks = ingest_pdf(pdf_path)
    records = app.state.embedder.embed_chunks(chunks)
    app.state.vector_store.upsert(records)

    state[pdf_path] = h
    _save_state(state)
