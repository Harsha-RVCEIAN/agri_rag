"""
glossary.py

Agricultural glossary and normalization rules.

PURPOSE:
- Correct common ASR (speech-to-text) errors
- Normalize crop names, fertilizers, schemes
- Preserve domain meaning before retrieval

DESIGN PRINCIPLES:
- Deterministic (rule-based only)
- Conservative (never over-correct)
- English is the INTERNAL language
- No translation responsibility

IMPORTANT:
- This module does NOT guess
- Applied AFTER translation to English
- Applied BEFORE embedding / retrieval
"""

from __future__ import annotations

import re
from typing import Dict


# ============================================================
# NORMALIZATION MAPS
# ============================================================

# ---------------- FERTILIZERS ----------------
FERTILIZER_MAP: Dict[str, str] = {
    r"\bdap\b": "DAP (Diammonium Phosphate)",
    r"\burea\b": "Urea",
    r"\bmop\b": "MOP (Muriate of Potash)",
    r"\bssp\b": "SSP (Single Super Phosphate)",
    r"\bnpk\b": "NPK fertilizer",
}

# ---------------- CROPS ----------------
CROP_MAP: Dict[str, str] = {
    r"\bpaddy\b": "rice",
    r"\brice crop\b": "rice",
    r"\bwheat crop\b": "wheat",
    r"\bmaize crop\b": "maize",
    r"\bcotton crop\b": "cotton",
    r"\btomato crop\b": "tomato",
}

# ---------------- UNITS / QUANTITIES ----------------
UNIT_MAP: Dict[str, str] = {
    r"\bkg\b": "kilogram",
    r"\bkgs\b": "kilograms",
    r"\bquintal\b": "quintal",
    r"\bqtl\b": "quintal",
    r"\bhectare\b": "hectare",
    r"\bha\b": "hectare",
    r"\bacre\b": "acre",
}

# ---------------- SCHEMES / POLICY ----------------
SCHEME_MAP: Dict[str, str] = {
    r"\bpmfby\b": "Pradhan Mantri Fasal Bima Yojana",
    r"\bpm kisan\b": "PM-KISAN scheme",
    r"\bpmksy\b": "Pradhan Mantri Krishi Sinchai Yojana",
    r"\bsoil health card\b": "Soil Health Card scheme",
}

# ---------------- COMMON ASR NOISE ----------------
ASR_NOISE_PATTERNS = (
    r"\bplease\b",
    r"\bkindly\b",
    r"\btell me\b",
    r"\bcan you\b",
    r"\bi want to know\b",
    r"\bwhat is about\b",
)


# ============================================================
# PUBLIC API
# ============================================================

def apply_glossary(text: str) -> str:
    """
    Apply glossary normalization rules to text.

    Rules:
    - Case-insensitive
    - Word-boundary safe
    - Non-destructive (text length may increase, not shrink)
    """

    if not text or not text.strip():
        return text

    normalized = text

    # ---------------- REMOVE ASR NOISE ----------------
    for pattern in ASR_NOISE_PATTERNS:
        normalized = re.sub(
            pattern,
            "",
            normalized,
            flags=re.IGNORECASE,
        )

    # ---------------- APPLY DOMAIN MAPS ----------------
    normalized = _apply_map(normalized, FERTILIZER_MAP)
    normalized = _apply_map(normalized, CROP_MAP)
    normalized = _apply_map(normalized, UNIT_MAP)
    normalized = _apply_map(normalized, SCHEME_MAP)

    # ---------------- WHITESPACE CLEANUP ----------------
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _apply_map(text: str, mapping: Dict[str, str]) -> str:
    """
    Apply regex-based replacements safely.
    """
    for pattern, replacement in mapping.items():
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE,
        )
    return text
