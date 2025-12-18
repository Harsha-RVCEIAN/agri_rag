import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from rag.pipeline import RAGPipeline
from ingestion.pipeline import ingest_pdf
from embeddings.embedder import Embedder
from embeddings.vector_store import VectorStore


# ---------------- APP INIT ----------------

app = FastAPI(
    title="Agri-RAG API",
    description="Testing API for document-grounded agricultural RAG system",
    version="1.0"
)

rag = RAGPipeline()
embedder = Embedder()
vector_store = VectorStore()

PDF_DIR = os.path.join("data", "pdfs")
os.makedirs(PDF_DIR, exist_ok=True)


# ---------------- SCHEMAS ----------------

class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    citations: list
    refused: bool
    refusal_reason: str | None = None
    diagnostics: dict | None = None


# ---------------- ROUTES ----------------

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Agri-RAG API running"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = rag.run(query=req.question)

    return {
        "answer": result["answer"],
        "confidence": result["confidence"],
        "citations": result["citations"],
        "refused": result["refused"],
        "refusal_reason": result.get("refusal_reason"),
        "diagnostics": result.get("diagnostics"),
    }


@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    pdf_path = os.path.join(PDF_DIR, file.filename)

    # ---- save file ----
    with open(pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ---- ingestion ----
    chunks = ingest_pdf(pdf_path)

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No usable content extracted from PDF"
        )

    embedded = embedder.embed_chunks(chunks)
    vector_store.upsert(embedded)

    return {
        "status": "success",
        "file": file.filename,
        "chunks_ingested": len(chunks)
    }
