"""
query_normalizer.py

Purpose:
- Normalize user queries BEFORE embedding & retrieval
- Improve recall for agriculture-related RAG
- Handle short, noisy, code-mixed farmer input
- Be SAFE for both text and voice pipelines

This module:
- DOES NOT translate
- DOES NOT detect language
- DOES NOT use LLMs
"""

from __future__ import annotations

import re
from typing import Dict, List

from nlp.glossary import apply_glossary


# ============================================================
# AGRICULTURE KEYWORDS (DOMAIN SIGNALS)
# ============================================================

AGRICULTURE_KEYWORDS = set([
    # --- CORE ---
    "agri","agriculture","farm","farming","farmer","crop","crops","soil","water",
    "irrigation","fertilizer","pest","disease","weed","seed","harvest","yield",
    "market","mandi","msp","livestock","dairy","horticulture",

    # --- MAJOR CROPS ---
    "rice","wheat","maize","millet","sorghum","barley",
    "soybean","groundnut","chickpea","lentil","mustard","sunflower",

    # --- VEGETABLES ---
    "tomato","onion","potato","brinjal","chili","okra","cabbage","cauliflower",

    # --- FRUITS ---
    "mango","banana","apple","grape","orange","coconut",

    # --- INPUTS ---
    "urea","dap","npk","compost","vermicompost","manure",
    "insecticide","pesticide","fungicide","herbicide",

    # --- WEATHER ---
    "rainfall","rain","drought","flood","temperature","climate",

    # --- PRACTICES ---
    "organic","natural","irrigation","drip","sprinkler",
    "crop","rotation","intercropping","greenhouse","polyhouse",

    # --- POLICY ---
    "scheme","insurance","subsidy","pmfby","pm kisan"
])


# ============================================================
# STOPWORDS (VOICE-SAFE, CONSERVATIVE)
# ============================================================

STOPWORDS = set([
    "what","is","are","the","a","an","of","for","to","in","on",
    "today","now","current","latest","please","tell","me",
    "give","explain","about","can","i","we","you","sir","bhai","bhaiya"
])


# ============================================================
# SYNONYM NORMALIZATION (STRICT)
# ============================================================

SYNONYM_MAP = {
    "paddy": "rice",
    "corn": "maize",
    "chilies": "chili",
    "fert": "fertilizer",
    "fertilizers": "fertilizer",
    "rainfed": "rainfall",
    "mandi": "market",
    "price": "market",
}


# ============================================================
# ASR / SPOKEN NOISE
# ============================================================

ASR_FILLER_PATTERNS = (
    r"\bplease\b",
    r"\btell me\b",
    r"\bcan you\b",
    r"\bi want to know\b",
    r"\bwhat about\b",
    r"\bhow much\b",
    r"\bhow many\b",
)


# ============================================================
# PUBLIC API
# ============================================================

def normalize_query(text: str) -> Dict[str, object]:
    """
    Normalize user query for retrieval.

    Input:
        English text (typed OR speech transcript)

    Returns:
    {
        "normalized_query": str,
        "keywords_found": List[str],
        "is_agriculture_related": bool
    }
    """

    if not text or not text.strip():
        return _empty_result()

    # ---------------- LOWER ----------------
    text = text.lower().strip()

    # ---------------- REMOVE ASR FILLERS ----------------
    for pattern in ASR_FILLER_PATTERNS:
        text = re.sub(pattern, " ", text)

    # ---------------- BASIC CLEAN ----------------
    cleaned = _basic_clean(text)

    # ---------------- GLOSSARY NORMALIZATION ----------------
    # 🔑 fixes DAP, PMFBY, units, crop names
    cleaned = apply_glossary(cleaned)

    # ---------------- TOKENIZE ----------------
    tokens = cleaned.split()

    # ---------------- SYNONYM NORMALIZATION ----------------
    tokens = [SYNONYM_MAP.get(tok, tok) for tok in tokens]

    # ---------------- STOPWORD REMOVAL ----------------
    filtered_tokens = [
        tok for tok in tokens
        if tok not in STOPWORDS and len(tok) > 1
    ]

    # ---------------- DOMAIN SIGNAL ----------------
    keywords_found = sorted({
        tok for tok in filtered_tokens
        if tok in AGRICULTURE_KEYWORDS
    })

    is_agriculture_related = len(keywords_found) > 0

    normalized_query = " ".join(filtered_tokens)

    return {
        "normalized_query": normalized_query,
        "keywords_found": keywords_found,
        "is_agriculture_related": is_agriculture_related,
    }


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _basic_clean(text: str) -> str:
    """
    Light cleaning only.
    DO NOT over-clean (important for short voice queries).
    """
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _empty_result() -> Dict[str, object]:
    return {
        "normalized_query": "",
        "keywords_found": [],
        "is_agriculture_related": False,
    }
