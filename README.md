# 🌾 Agri-RAG: Agricultural Information System with Intelligent Fallback

Agri-RAG is a **domain-restricted agricultural question answering system** built using **Retrieval-Augmented Generation (RAG)**.  
It prioritizes **document-grounded answers** from an agricultural knowledge base and transparently falls back to a **general-purpose AI model** only when required information is unavailable.

This project focuses on **system design, reliability, and responsible AI usage**, rather than training large language models.

---

## 🚀 Key Features

- ✅ Domain-specific (Agriculture-only) query handling  
- 📄 PDF ingestion with OCR, table extraction, and chunking  
- 🔎 Semantic retrieval using vector embeddings (MiniLM + Pinecone)  
- 🤖 Grounded answer generation using FLAN-T5-Base (CPU-friendly)  
- ⚠️ Transparent external LLM fallback (Gemini Pro API)  
- 🧠 Intelligent routing based on retrieval confidence  
- 🛑 Hallucination prevention through strict grounding rules  
- 🌐 REST API built with FastAPI  

---

## 🧩 System Architecture

User Query
↓
Domain Check (Agriculture?)
↓
Query Embedding (MiniLM)
↓
Vector Retrieval (Pinecone)
↓
Retrieval Confidence Evaluation
├──  RAG Answer
---

## 🛠️ Technology Stack

| Component | Technology |
|---------|------------|
| Backend API | FastAPI |
| Embeddings | sentence-transformers (MiniLM) |
| Vector DB | Pinecone |
| Local LLM | FLAN-T5-Base |
| Fallback LLM | Gemini Pro API (optional) |
| OCR | Tesseract |
| PDF Parsing | pdfplumber |
| Language | Python |

---

## 📂 Project Structure

agri_rag/
├── api/ # FastAPI backend
├── ingestion/ # PDF ingestion & OCR pipeline
├── embeddings/ # Embedding & vector store logic
├── rag/ # Retrieval & scoring logic
├── llm/ # LLM clients and answer generation
├── data/ # PDFs and vector store state
├── scripts/ # CLI & utility scripts
├── tests/ # Test cases
├── frontend/ # Simple web UI
├── config.py
├── requirements.txt
└── README.md

--

agri_rag/
├── api/ # FastAPI backend
├── ingestion/ # PDF ingestion & OCR pipeline
├── embeddings/ # Embedding & vector store logic
├── rag/ # Retrieval & scoring logic
├── llm/ # LLM clients and answer generation
├── data/ # PDFs and vector store state
├── scripts/ # CLI & utility scripts
├── tests/ # Test cases
├── frontend/ # Simple web UI
├── config.py
├── requirements.txt
└── README.md

--