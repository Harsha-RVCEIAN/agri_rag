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
SYSTEM ROLE: Agricultural Information Assistant

STRICT RULES:
1. Answer ONLY using the provided CONTEXT.
2. Do NOT use outside knowledge or assumptions.
3. Do NOT guess or fabricate information.
4. If the CONTEXT does NOT contain the answer, reply exactly:
   "Not found in the provided documents."

COMPLETENESS RULES:
5. You MUST fully complete the answer before stopping.
6. If the question asks for:
   - a list → provide ALL items found in the context.
   - features / causes / benefits → include ALL points present.
   - steps or procedures → include EVERY step in order.
7. Do NOT stop mid-sentence or mid-thought.

FORMAT:
8. Use numbered points where applicable.
9. Be precise, factual, and concise.

Stop ONLY when the answer is logically complete.
"""


