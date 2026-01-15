import re
from typing import Dict, Optional


# =========================================================
# CONFIG THRESHOLDS (DEFAULTS)
# =========================================================

MIN_TEXT_LENGTH = 40
MIN_ALPHA_RATIO = 0.35
MAX_HEADER_REPEAT_SCORE = 0.6

MIN_IMAGE_COUNT_FOR_OCR = 2   # 🔑 avoid logo-triggered OCR

MIN_MEANINGFUL_WORDS = 6
MAX_DIGIT_RATIO = 0.4

# table hint: numbers + units often appear together
TABLE_UNIT_HINTS = r"\b(kg|g|quintal|ha|acre|%|rs|₹)\b"


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def _alphanumeric_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(c.isalpha() for c in text)
    return alpha / max(len(text), 1)


def _digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    digits = sum(c.isdigit() for c in text)
    return digits / max(len(text), 1)


def _meaningful_word_count(text: str) -> int:
    return len(re.findall(r"[a-zA-Z]{3,}", text))


def _looks_like_header_or_footer(text: str, repeat_score: float) -> bool:
    if not text:
        return True

    t = text.strip()
    t_lower = t.lower()

    # ---------- repetition across pages ----------
    if repeat_score > MAX_HEADER_REPEAT_SCORE:
        return True

    # ---------- page numbering ----------
    if re.fullmatch(r"(page\s*)?\d+(\s*/\s*\d+)?", t_lower):
        return True

    # ---------- boilerplate ----------
    boilerplate_patterns = [
        r"\bgovernment of\b",
        r"\bministry of\b",
        r"\bdepartment of\b",
        r"\bभारत सरकार\b",
        r"\bgovt\.?\b",
        r"\bwww\.",
        r"\bhttp[s]?://",
    ]
    if any(re.search(p, t_lower) for p in boilerplate_patterns) and len(t) < 150:
        return True

    # ---------- all-caps short ----------
    if len(t) < 60 and t.isupper():
        return True

    # ---------- lexical repetition ----------
    words = re.findall(r"\b\w+\b", t_lower)
    if words:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.45 and len(words) < 20:
            return True

    return False


def _image_dominant(page: Dict) -> bool:
    return (
        len(page.get("images", [])) >= MIN_IMAGE_COUNT_FOR_OCR
        and len(page.get("text", "").strip()) < MIN_TEXT_LENGTH
    )


def _looks_like_table(text: str) -> bool:
    """
    Numeric-heavy BUT structured → likely table, not junk.
    """
    if not text:
        return False

    if re.search(TABLE_UNIT_HINTS, text.lower()):
        return True

    lines = text.splitlines()
    numeric_lines = sum(
        1 for l in lines if _digit_ratio(l) > 0.3 and len(l.split()) >= 2
    )

    return numeric_lines >= max(2, len(lines) // 3)


# =========================================================
# MAIN CLASSIFIER
# =========================================================

def classify_page(
    page: Dict,
    doc_rules: Optional[Dict] = None
) -> str:
    """
    Conservative page classifier.

    Returns:
    - TEXT_OK
    - OCR_REQUIRED

    doc_rules allows SAFE overrides per document class.
    """

    doc_rules = doc_rules or {}
    domain = doc_rules.get("domain")

    text = page.get("text", "").strip()
    repeat_score = page.get("header_repeat_score", 0.0)

    # =====================================================
    # 🔑 DOMAIN OVERRIDES (STRICT & LIMITED)
    # =====================================================

    # Statistics / reports → NEVER OCR
    if domain in {"statistics", "market"}:
        return "TEXT_OK" if text else "OCR_REQUIRED"

    # Schemes → text preferred, OCR discouraged
    if domain == "scheme":
        if text and _meaningful_word_count(text) >= 4:
            return "TEXT_OK"

    # =====================================================
    # DEFAULT CLASSIFICATION LOGIC
    # =====================================================

    # ---------- no text layer ----------
    if not text:
        return "OCR_REQUIRED"

    # ---------- image-dominant ----------
    if _image_dominant(page):
        return "OCR_REQUIRED"

    # ---------- headers / footers ----------
    if _looks_like_header_or_footer(text, repeat_score):
        return "OCR_REQUIRED"

    # ---------- junk symbols ----------
    if _alphanumeric_ratio(text) < MIN_ALPHA_RATIO:
        return "OCR_REQUIRED"

    # ---------- numeric-heavy ----------
    if _digit_ratio(text) > MAX_DIGIT_RATIO and not _looks_like_table(text):
        return "OCR_REQUIRED"

    # ---------- too little semantic content ----------
    if _meaningful_word_count(text) < MIN_MEANINGFUL_WORDS:
        return "OCR_REQUIRED"

    return "TEXT_OK"
