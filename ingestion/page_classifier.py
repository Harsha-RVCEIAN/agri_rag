# ingestion/page_classifier.py

import re
from typing import Dict

# ---------------- CONFIG THRESHOLDS ----------------

MIN_TEXT_LENGTH = 30          # below this → probably header-only
MIN_ALPHA_RATIO = 0.3         # too many symbols/numbers = junk
MAX_HEADER_REPEAT_SCORE = 0.6 # repeated headers across pages
MIN_IMAGE_COUNT_FOR_OCR = 1   # image-dominant page


# ---------------- HELPER FUNCTIONS ----------------

def _alphanumeric_ratio(text: str) -> float:
    """Ratio of alphabetic characters to total length."""
    if not text:
        return 0.0
    alpha = sum(c.isalpha() for c in text)
    return alpha / max(len(text), 1)


def _looks_like_header_or_footer(
    text: str,
    repeat_score: float = 0.0
) -> bool:
    """
    Robust header/footer detector using multiple signals.
    """

    if not text:
        return True

    t = text.strip()
    t_lower = t.lower()

    # ---------- Strong signal: repetition across pages ----------
    if repeat_score > MAX_HEADER_REPEAT_SCORE:
        return True

    # ---------- Page number patterns ----------
    page_patterns = [
        r"^page\s+\d+",
        r"^\d+\s*/\s*\d+$",
        r"^\d+$"
    ]
    for pat in page_patterns:
        if re.fullmatch(pat, t_lower):
            return True

    # ---------- Government boilerplate ----------
    boilerplate_patterns = [
        r"\bgovernment of\b",
        r"\bministry of\b",
        r"\bdepartment of\b",
        r"\bभारत सरकार\b",
        r"\bgovt\.?\b",
        r"\bwww\.",
        r"\bhttp[s]?://"
    ]
    for pat in boilerplate_patterns:
        if re.search(pat, t_lower) and len(t) < 120:
            return True

    # ---------- All-caps short lines ----------
    if len(t) < 50 and t.isupper():
        return True

    # ---------- Low lexical diversity ----------
    words = re.findall(r"\b\w+\b", t_lower)
    if words:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.4 and len(t) < 120:
            return True

    return False


def _image_dominant(page: Dict) -> bool:
    """
    Page is image-dominant if it contains images
    AND very little usable text.
    """
    return (
        len(page.get("images", [])) >= MIN_IMAGE_COUNT_FOR_OCR
        and len(page.get("text", "").strip()) < MIN_TEXT_LENGTH
    )


# ---------------- MAIN CLASSIFIER ----------------

def classify_page(page: Dict) -> str:
    """
    Classify a page into:
    - TEXT_OK        → clean text, no OCR
    - OCR_REQUIRED   → scanned or weak-text page
    """

    text = page.get("text", "").strip()
    repeat_score = page.get("header_repeat_score", 0.0)

    # ---------- CASE 1: No text layer ----------
    if not text:
        return "OCR_REQUIRED"

    # ---------- CASE 2: Image-dominant page ----------
    if _image_dominant(page):
        return "OCR_REQUIRED"

    # ---------- CASE 3: Junk / non-semantic text ----------
    if _alphanumeric_ratio(text) < MIN_ALPHA_RATIO:
        return "OCR_REQUIRED"

    # ---------- CASE 4: Header/footer only ----------
    if _looks_like_header_or_footer(text, repeat_score):
        return "OCR_REQUIRED"

    # ---------- CASE 5: Clean text ----------
    return "TEXT_OK"
