"""
Shared utilities for all ingestion transformers.

Responsibilities:
- Drop junk sections
- Drop weak pages
- Normalize text
- Detect headings
- Apply global ingestion guards

This file MUST:
- Be deterministic
- Contain NO class-specific logic
- Contain NO embeddings / OCR / tables
"""

import re
from typing import List, Dict


# =========================================================
# GLOBAL CONFIG (APPLIES TO ALL CLASSES)
# =========================================================

MIN_TEXT_CHARS = 250
MAX_TEXT_CHARS = 6000

DROP_SECTION_KEYWORDS = [
    "acknowledgement",
    "acknowledgment",
    "preface",
    "foreword",
    "references",
    "bibliography",
    "annexure",
    "appendix",
    "glossary",
    "abbreviations",
    "copyright",
    "disclaimer",
    "index",
]

DROP_LINE_KEYWORDS = [
    "all rights reserved",
    "printed by",
    "published by",
    "isbn",
]

HEADING_MAX_LENGTH = 120   # 🔑 widened safely


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================
# SECTION / LINE FILTERING
# =========================================================

def should_drop_section(title: str) -> bool:
    if not title:
        return False

    t = title.lower()
    return any(k in t for k in DROP_SECTION_KEYWORDS)


def should_drop_line(line: str) -> bool:
    """
    Drop boilerplate lines but preserve short semantic labels.
    """
    if not line:
        return True

    l = line.lower().strip()

    # pure noise
    if not any(c.isalpha() for c in l):
        return True

    return any(k in l for k in DROP_LINE_KEYWORDS)


# =========================================================
# HEADING DETECTION
# =========================================================

def looks_like_heading(line: str) -> bool:
    if not line:
        return False

    line = line.strip()

    if len(line) > HEADING_MAX_LENGTH:
        return False

    # ALL CAPS
    if line.isupper():
        return True

    # Ends with colon
    if line.endswith(":"):
        return True

    # Numbered headings
    if re.match(r"^\d+(\.\d+)*[\).\s]+[A-Za-z ]+$", line):
        return True

    return False


# =========================================================
# PAGE-LEVEL FILTERS
# =========================================================

def is_usable_page(page: Dict) -> bool:
    text = page.get("text", "")
    if not text:
        return False

    text = normalize_text(text)

    if len(text) < MIN_TEXT_CHARS:
        return False

    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    if digit_ratio > 0.5:
        return False

    return True


# =========================================================
# SECTION BUILDER
# =========================================================

def split_into_sections(text: str) -> List[Dict]:
    sections: List[Dict] = []

    current_title = "general"
    buffer: List[str] = []

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    def flush():
        nonlocal buffer, current_title

        if current_title == "dropped":
            buffer = []
            return

        content = normalize_text(" ".join(buffer))
        buffer = []

        if len(content) < MIN_TEXT_CHARS:
            return

        # numeric-heavy section guard
        digit_ratio = sum(c.isdigit() for c in content) / max(len(content), 1)
        if digit_ratio > 0.6:
            return

        sections.append({
            "title": current_title,
            "content": content,
        })

    for line in lines:
        if looks_like_heading(line):
            title = line.rstrip(":").strip()

            if should_drop_section(title):
                buffer = []
                current_title = "dropped"
                continue

            flush()
            current_title = title
            continue

        if should_drop_line(line):
            continue

        buffer.append(line)

    flush()
    return sections


# =========================================================
# FINAL SAFETY
# =========================================================

def enforce_length_limits(text: str) -> str:
    if not text:
        return ""

    if len(text) > MAX_TEXT_CHARS:
        return text[:MAX_TEXT_CHARS]

    return text
