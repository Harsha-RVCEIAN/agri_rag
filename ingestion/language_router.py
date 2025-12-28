from typing import Dict
from langdetect import detect_langs, LangDetectException
import re

# ---------------- CONFIG ----------------

SUPPORTED_LANGS = {
    "en": "eng",
    "hi": "hin",
    "ta": "tam",
    "te": "tel",
    "kn": "kan",
    "ml": "mal",
}

MIN_CONFIDENCE = 0.75
MIN_TEXT_LENGTH = 30
MIN_ALPHA_RATIO = 0.25

# Indian-language script hints (cheap, reliable)
SCRIPT_HINTS = {
    "hi": r"[अ-ह]",
    "ta": r"[அ-ஹ]",
    "te": r"[అ-హ]",
    "kn": r"[ಅ-ಹ]",
    "ml": r"[അ-ഹ]",
}


# ---------------- HELPERS ----------------

def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(c.isalpha() for c in text)
    return alpha / max(len(text), 1)


def _script_override(text: str) -> str | None:
    """
    Script detection beats probabilistic detection.
    """
    for lang, pattern in SCRIPT_HINTS.items():
        if re.search(pattern, text):
            return lang
    return None


# ---------------- CORE ----------------

def route_language(text: str) -> Dict:
    """
    Detect language and return routing decision.

    Returns:
    {
        language: en | hi | ta | te | kn | ml | mixed | unknown
        ocr_model: eng | hin | eng+hin | None
        confidence: float
    }
    """

    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return {
            "language": "unknown",
            "ocr_model": None,
            "confidence": 0.0,
        }

    # Garbage / numeric / OCR-noise guard
    if _alpha_ratio(text) < MIN_ALPHA_RATIO:
        return {
            "language": "unknown",
            "ocr_model": None,
            "confidence": 0.0,
        }

    # ---------- SCRIPT OVERRIDE ----------
    script_lang = _script_override(text)
    if script_lang and script_lang in SUPPORTED_LANGS:
        return {
            "language": script_lang,
            "ocr_model": SUPPORTED_LANGS[script_lang],
            "confidence": 0.99,  # script certainty
        }

    # ---------- LANGDETECT ----------
    try:
        detections = detect_langs(text)
    except LangDetectException:
        return {
            "language": "unknown",
            "ocr_model": None,
            "confidence": 0.0,
        }

    if not detections:
        return {
            "language": "unknown",
            "ocr_model": None,
            "confidence": 0.0,
        }

    top = detections[0]
    lang = top.lang
    conf = top.prob

    # ---------- MIXED LANGUAGE ----------
    if len(detections) > 1:
        second = detections[1]
        if conf < MIN_CONFIDENCE and second.prob > 0.25:
            return {
                "language": "mixed",
                "ocr_model": "eng+hin",
                "confidence": round(conf, 3),
            }

    # ---------- UNSUPPORTED ----------
    if lang not in SUPPORTED_LANGS:
        return {
            "language": "unknown",
            "ocr_model": None,
            "confidence": round(conf, 3),
        }

    # ---------- SUPPORTED ----------
    return {
        "language": lang,
        "ocr_model": SUPPORTED_LANGS[lang],
        "confidence": round(conf, 3),
    }
