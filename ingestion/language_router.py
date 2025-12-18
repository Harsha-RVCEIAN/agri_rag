# ingestion/language_router.py

from typing import Dict
from langdetect import detect_langs, LangDetectException


# ---------------- CONFIG ----------------

# Languages you actually support in OCR / embeddings
SUPPORTED_LANGS = {
    "en": "eng",
    "hi": "hin",
    "ta": "tam",
    "te": "tel",
    "kn": "kan",
    "ml": "mal",
}

# If no language crosses this probability → mixed
MIN_CONFIDENCE = 0.75


# ---------------- CORE LOGIC ----------------

def route_language(text: str) -> Dict:
    """
    Detect language of text and return routing info.

    Returns:
    {
        "language": "en | hi | ta | te | kn | ml | mixed | unknown",
        "ocr_model": "eng | hin | eng+hin | None",
        "confidence": float
    }
    """

    if not text or len(text.strip()) < 20:
        return {
            "language": "unknown",
            "ocr_model": None,
            "confidence": 0.0
        }

    try:
        detections = detect_langs(text)
    except LangDetectException:
        return {
            "language": "unknown",
            "ocr_model": None,
            "confidence": 0.0
        }

    # ---- No detections ----
    if not detections:
        return {
            "language": "unknown",
            "ocr_model": None,
            "confidence": 0.0
        }

    # ---- Best detection ----
    top = detections[0]
    lang = top.lang
    conf = top.prob

    # ---- Mixed language detection ----
    if len(detections) > 1:
        second = detections[1]
        if conf < MIN_CONFIDENCE and second.prob > 0.2:
            return {
                "language": "mixed",
                "ocr_model": "eng+hin",  # safest fallback for India
                "confidence": conf
            }

    # ---- Unsupported language ----
    if lang not in SUPPORTED_LANGS:
        return {
            "language": "unknown",
            "ocr_model": None,
            "confidence": conf
        }

    # ---- Supported language ----
    return {
        "language": lang,
        "ocr_model": SUPPORTED_LANGS[lang],
        "confidence": conf
    }
