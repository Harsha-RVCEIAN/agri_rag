import os
import sys
import shutil
import hashlib
import re
from typing import Optional, Dict, List

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------- PATH SETUP ----------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------- INTERNAL IMPORTS ----------------

from rag.retriever import Retriever
from llm.answerer import generate_answer
from llm.llm_client import LLMClient
from ingestion.pipeline import ingest_pdf
from embeddings.embedder import Embedder
from embeddings.vector_store import VectorStore

# ---------------- APP INIT ----------------

app = FastAPI(
    title="Agri-RAG API",
    description="Agriculture-only RAG with safe Gemini fallback",
    version="1.0.0",
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

# ---------------- STARTUP ----------------

@app.on_event("startup")
def startup_event():
    app.state.embedder = Embedder()
    app.state.vector_store = VectorStore()
    app.state.retriever = Retriever(app.state.vector_store)
    app.state.gemini_llm = LLMClient(provider="gemini")
    print("✅ Agri-RAG startup complete")

# ---------------- DOMAIN GUARDS ----------------

AGRI_KEYWORDS = {
    "agriculture", "farming", "crop", "soil", "fertilizer",
    "rice", "wheat", "maize", "paddy", "tomato", "potato",
    "organic", "irrigation", "yield", "harvest",
    "scheme", "pm kisan", "pmfby", "insurance",
}

PERSON_PATTERNS = [
    r"\bwho is\b",
    r"\bbiography\b",
    r"\bborn\b",
    r"\bage\b",
    r"\bnet worth\b",
]


def is_agriculture_query(q: str) -> bool:
    q = q.lower()
    return any(k in q for k in AGRI_KEYWORDS)


def looks_like_person_query(q: str) -> bool:
    q = q.lower()
    return any(re.search(p, q) for p in PERSON_PATTERNS)

# ---------------- MULTI QUESTION SPLIT ----------------

def split_questions(text: str) -> List[str]:
    parts = re.split(r"\band\b|\balso\b|;", text, flags=re.I)
    return [p.strip() for p in parts if len(p.split()) > 2]

# ---------------- UTILS ----------------

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

# ---------------- SCHEMAS ----------------

class ChatRequest(BaseModel):
    question: str
    intent: Optional[str] = None
    language: Optional[str] = None
    top_k: int = 6


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    citations: List[Dict]
    refused: bool
    refusal_reason: Optional[str] = None
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

    # ---- PERSON BLOCK ----
    if looks_like_person_query(raw):
        return ChatResponse(
            answer="This system answers agriculture-related questions only.",
            confidence=0.0,
            citations=[],
            refused=True,
            refusal_reason="person_query"
        )

    # ---- NON AGRI BLOCK ----
    if not is_agriculture_query(raw):
        return ChatResponse(
            answer="This system answers agriculture-related questions only.",
            confidence=0.0,
            citations=[],
            refused=True,
            refusal_reason="non_agriculture"
        )

    questions = split_questions(raw)

    answers = []
    confidence = 0.0

    for q in questions:
        vectors = app.state.embedder.embed_texts([q])
        retrieval = app.state.retriever.retrieve(
            query=q,
            query_vectors=vectors,
            intent=req.intent,
            language=req.language,
            top_k=req.top_k,
        )

        chunks = retrieval["chunks"]

        # ---- NO DOCS → GEMINI ----
        if not chunks:
            fallback = app.state.gemini_llm.generate(
                system_prompt=(
                    "You are an agricultural expert. "
                    "Answer clearly and concisely. "
                    "Do not mention documents or AI systems."
                ),
                user_prompt=q,
                temperature=0.2,
                max_tokens=180
            )
            answers.append(fallback)
            confidence = max(confidence, 0.3)
            continue

        rag = generate_answer(q, chunks, retrieval["diagnostics"])

        if not rag["answer"]:
            fallback = app.state.gemini_llm.generate(
                system_prompt="You are an agricultural expert.",
                user_prompt=q,
                temperature=0.2,
                max_tokens=180
            )
            answers.append(fallback)
            confidence = max(confidence, 0.3)
            continue

        answers.append(rag["answer"])
        confidence = max(confidence, max(c["score"] for c in chunks))

    return ChatResponse(
        answer="\n\n".join(answers),
        confidence=min(1.0, confidence),
        citations=[],   # 🔒 hidden by design
        refused=False,
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
