# config.py

# ---------------- PROJECT ----------------
PROJECT_NAME = "Agri-Connect RAG"
VERSION = "1.0.0"


# ---------------- PDF / OCR ----------------
PDF_TEXT_MIN_LENGTH = 30        # below this → OCR
MIN_ALPHA_RATIO = 0.3           # junk-text threshold

TARGET_OCR_DPI = 300
MAX_SKEW_ANGLE = 5.0

SUPPORTED_LANGUAGES = {
    "eng": "English",
    "hin": "Hindi",
    "kan": "Kannada",
    "tel": "Telugu",
    "tam": "Tamil",
    "mal": "Malayalam"
}

DEFAULT_OCR_LANGUAGE = "eng"
LOW_OCR_CONFIDENCE_THRESHOLD = 0.6


# ---------------- CHUNKING ----------------
MAX_CHUNK_LENGTH = 400          # characters
MIN_CHUNK_LENGTH = 40


# ---------------- EMBEDDINGS ----------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


# ---------------- VECTOR DATABASE ----------------
VECTOR_DB_COLLECTION = "agri_knowledge"
VECTOR_TOP_K = 6


# ---------------- RAG / RETRIEVAL ----------------
PREFER_DIGITAL_TEXT = True
DEMOTE_LOW_CONFIDENCE_OCR = True

MIN_ACCEPTABLE_CONTEXT_CONFIDENCE = 0.7


# ---------------- PROMPTING ----------------
SYSTEM_PROMPT = """
You are an agriculture assistant.
Answer ONLY using the provided context.
If the context is insufficient, unclear, or low confidence,
say you do not know and ask for clarification.
Do NOT guess.
"""

