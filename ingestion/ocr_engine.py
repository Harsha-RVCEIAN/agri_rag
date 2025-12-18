# ingestion/ocr_engine.py

import pytesseract
from pytesseract import Output
import numpy as np
from typing import Tuple, Dict, List


# ---------------- CONFIG ----------------

# Mapping between detected language/script and Tesseract models
LANG_MODEL_MAP = {
    "eng": "eng",
    "hin": "hin",
    "kan": "kan",
    "tel": "tel",
    "tam": "tam",
    "mal": "mal",
    "mixed": "eng+hin"  # safe fallback
}

# Words below this confidence are considered unreliable
MIN_WORD_CONFIDENCE = 40


# ---------------- CORE OCR ----------------

def run_ocr(
    image: np.ndarray,
    language: str = "eng"
) -> Tuple[str, float, Dict]:
    """
    Run OCR on a preprocessed image.

    Returns:
        text (str): extracted OCR text
        confidence (float): page-level confidence [0–1]
        details (dict): block/word-level OCR metadata
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

    for i in range(len(ocr_data["text"])):
        word = ocr_data["text"][i].strip()
        conf = ocr_data["conf"][i]

        if not word:
            continue

        try:
            conf = int(conf)
        except ValueError:
            continue

        if conf < 0:
            continue

        words.append(word)
        confidences.append(conf)

    # -------- Text assembly --------
    text = " ".join(words)

    # -------- Confidence aggregation --------
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        page_confidence = avg_conf / 100.0
    else:
        page_confidence = 0.0

    # -------- Detailed metadata --------
    details = {
        "word_count": len(words),
        "avg_word_confidence": page_confidence,
        "language_model": lang_model,
        "raw_ocr_data": ocr_data
    }

    return text.strip(), page_confidence, details
