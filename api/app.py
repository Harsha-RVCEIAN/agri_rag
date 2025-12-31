from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=True)

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
    version="2.2.2",
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

def _save_state(state: Dict[str, str]) -> None:
    import json
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

# ---------------- STARTUP ----------------

@app.on_event("startup")
def startup_event():
    embedder = Embedder()
    vector_store = VectorStore()

    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.rag = RAGPipeline(
        vector_store=vector_store,
        embedder=embedder
    )
    # 🔥 SINGLE Gemini instance
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
    return any(k in q.lower() for k in AGRI_KEYWORDS)

def looks_like_person_query(q: str) -> bool:
    return any(re.search(p, q.lower()) for p in PERSON_PATTERNS)

def is_definition_query(q: str) -> bool:
    q = q.lower().strip()
    return any(q.startswith(t) for t in DEFINITION_TRIGGERS)

# ---------------- RAG CACHE ----------------

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

    # ================= DEFINITIONS (GEMINI ONLY) =================
    if is_definition_query(raw):
        answer = app.state.gemini_llm.generate(
            system_prompt=(
                "You are an agricultural expert.\n"
                "Give a clear, textbook-style definition.\n"
                "No advice. No recommendations."
            ),
            user_prompt=raw,
            temperature=0.3,
            max_tokens=500,
        )

        # 🔥 NEVER ERROR — ALWAYS RETURN TEXT
        return ChatResponse(
            status="answer",
            answer=answer or "Definition not available at the moment.",
            confidence=0.7,
            diagnostics={"category": "definition"}
        )

    # ================= RAG =================
    result = cached_rag_answer(
        raw,
        req.intent,
        req.language,
        req.category
    )

    if result.get("status") == "answer":
        return ChatResponse(
            status="answer",
            answer=result.get("answer"),
            confidence=result.get("confidence", 0.0),
            diagnostics=result.get("diagnostics")
        )

    # ================= GEMINI FALLBACK =================
    fallback = app.state.gemini_llm.generate(
        system_prompt="You are an agricultural expert.",
        user_prompt=raw,
        temperature=0.3,
        max_tokens=350,
    )

    if fallback:
        return ChatResponse(
            status="answer",
            answer=fallback,
            confidence=0.6,
            diagnostics={"fallback": "gemini"}
        )

    return ChatResponse(
        status="no_answer",
        answer=None,
        confidence=0.0,
        message="Unable to answer with available information.",
        suggestion="Try rephrasing or upload relevant documents."
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
