"""
language_detector.py

Language detection for USER QUERIES (chat input).

Scope:
- Supports ALL 22 official languages of India
- Includes script metadata
- Safe for short, noisy, code-mixed input
- Deterministic and LLM-free

IMPORTANT:
- Detection reliability is NOT guaranteed for short queries
- If is_reliable == False → downstream must NOT translate blindly
"""

from typing import Dict
from langdetect import detect_langs, LangDetectException


# ============================================================
# OFFICIAL LANGUAGES OF INDIA (WITH SCRIPT)
# ============================================================

# ISO 639-1 where available (langdetect compatible)
OFFICIAL_LANGUAGES = {
    "as": {"name": "Assamese", "script": "Bengali–Assamese"},
    "bn": {"name": "Bengali", "script": "Bengali"},
    "en": {"name": "English", "script": "Latin"},
    "gu": {"name": "Gujarati", "script": "Gujarati"},
    "hi": {"name": "Hindi", "script": "Devanagari"},
    "kn": {"name": "Kannada", "script": "Kannada"},
    "ks": {"name": "Kashmiri", "script": "Arabic / Devanagari"},
    "kok": {"name": "Konkani", "script": "Devanagari"},
    "ml": {"name": "Malayalam", "script": "Malayalam"},
    "mr": {"name": "Marathi", "script": "Devanagari"},
    "ne": {"name": "Nepali", "script": "Devanagari"},
    "or": {"name": "Odia", "script": "Odia"},
    "pa": {"name": "Punjabi", "script": "Gurmukhi"},
    "sa": {"name": "Sanskrit", "script": "Devanagari"},
    "sd": {"name": "Sindhi", "script": "Arabic / Devanagari"},
    "ta": {"name": "Tamil", "script": "Tamil"},
    "te": {"name": "Telugu", "script": "Telugu"},
    "ur": {"name": "Urdu", "script": "Perso-Arabic"},
    "mai": {"name": "Maithili", "script": "Devanagari"},
    "bho": {"name": "Bhojpuri", "script": "Devanagari"},
    "mni": {"name": "Manipuri (Meitei)", "script": "Meitei Mayek"},
    "sat": {"name": "Santali", "script": "Ol Chiki"},
}


# ============================================================
# DETECTION RULES
# ============================================================

MIN_CONFIDENCE = 0.70
MIN_TEXT_LENGTH = 4   # below this → unreliable by definition


# ============================================================
# PUBLIC API
# ============================================================

def detect_language(text: str) -> Dict[str, object]:
    """
    Detect language of user input.

    Returns:
    {
        "language": "hi" | "en" | "ta" | "unknown",
        "language_name": "Hindi",
        "script": "Devanagari",
        "confidence": 0.93,
        "is_reliable": True
    }

    CONTRACT:
    - Never raises
    - Never fabricates confidence
    - 'unknown' forces downstream handling
    """

    # ---------------- BASIC SANITY ----------------
    if not text or not text.strip():
        return _unknown()

    if len(text.strip()) < MIN_TEXT_LENGTH:
        return _unknown()

    try:
        detections = detect_langs(text)

        if not detections:
            return _unknown()

        top = detections[0]
        lang_code = top.lang
        confidence = round(top.prob, 3)

        if confidence < MIN_CONFIDENCE:
            return _unknown(confidence)

        lang_meta = OFFICIAL_LANGUAGES.get(lang_code)

        # Language detected but not an official Indian language
        if not lang_meta:
            return _unknown(confidence)

        return {
            "language": lang_code,
            "language_name": lang_meta["name"],
            "script": lang_meta["script"],
            "confidence": confidence,
            "is_reliable": True,
        }

    except LangDetectException:
        return _unknown()

    except Exception:
        return _unknown()


# ============================================================
# INTERNAL
# ============================================================

def _unknown(confidence: float = 0.0) -> Dict[str, object]:
    return {
        "language": "unknown",
        "language_name": None,
        "script": None,
        "confidence": round(confidence, 3),
        "is_reliable": False,
    }
