import pytesseract
from pytesseract import Output
import numpy as np
from typing import Tuple, Dict, List
import re

# ---------------- CONFIG ----------------

LANG_MODEL_MAP = {
    "eng": "eng",
    "hin": "hin",
    "kan": "kan",
    "tel": "tel",
    "tam": "tam",
    "mal": "mal",
    "mixed": "eng+hin",
}

MIN_WORD_CONFIDENCE = 40
MIN_MEANINGFUL_WORDS = 5

MAX_OCR_CONFIDENCE = 0.75          # OCR is never authoritative
NOISE_PENALTY_RATIO = 0.35
NUMERIC_HEAVY_PENALTY = 0.7        # 🔑 numeric OCR is riskier


# ---------------- HELPERS ----------------

def _noise_ratio(text: str) -> float:
    if not text:
        return 1.0
    noise = len(re.findall(r"[^a-zA-Z0-9\s]", text))
    return noise / max(len(text), 1)


def _meaningful_word_count(words: List[str]) -> int:
    return len([w for w in words if len(w) >= 3 and w.isalpha()])


def _numeric_ratio(words: List[str]) -> float:
    if not words:
        return 1.0
    numeric = sum(any(c.isdigit() for c in w) for w in words)
    return numeric / len(words)


# ---------------- CORE OCR ----------------

def run_ocr(
    image: np.ndarray,
    language: str = "eng"
) -> Tuple[str, float, Dict]:
    """
    Conservative OCR extraction.

    Returns:
        text (str)
        confidence (float)   # pessimistic, capped
        details (dict)
    """

    lang_model = LANG_MODEL_MAP.get(language, "eng")

    ocr_data = pytesseract.image_to_data(
        image,
        lang=lang_model,
        output_type=Output.DICT,
        config="--oem 3 --psm 6"
    )

    words: List[str] = []
    confidences: List[int] = []

    for i in range(len(ocr_data.get("text", []))):
        word = ocr_data["text"][i].strip()
        conf = ocr_data["conf"][i]

        if not word:
            continue

        try:
            conf = int(conf)
        except Exception:
            continue

        if conf < MIN_WORD_CONFIDENCE:
            continue

        words.append(word)
        confidences.append(conf)

    text = " ".join(words)

    # ---------------- CONFIDENCE ----------------

    meaningful_words = _meaningful_word_count(words)

    if not confidences or meaningful_words < MIN_MEANINGFUL_WORDS:
        page_confidence = 0.0
    else:
        avg_conf = sum(confidences) / len(confidences)
        page_confidence = avg_conf / 100.0

        # noise penalty
        noise = _noise_ratio(text)
        if noise > NOISE_PENALTY_RATIO:
            page_confidence *= (1 - noise)

        # numeric-heavy penalty (tables / dosages / prices)
        numeric_ratio = _numeric_ratio(words)
        if numeric_ratio > 0.4:
            page_confidence *= NUMERIC_HEAVY_PENALTY

        # hard cap
        page_confidence = min(page_confidence, MAX_OCR_CONFIDENCE)

    page_confidence = round(page_confidence, 3)

    # ---------------- DETAILS ----------------

    details = {
        "word_count": len(words),
        "meaningful_words": meaningful_words,
        "avg_word_confidence": round(
            sum(confidences) / len(confidences) / 100.0, 3
        ) if confidences else 0.0,
        "noise_ratio": round(_noise_ratio(text), 3),
        "numeric_ratio": round(_numeric_ratio(words), 3),
        "language_model": lang_model,
    }

    return text.strip(), page_confidence, details
