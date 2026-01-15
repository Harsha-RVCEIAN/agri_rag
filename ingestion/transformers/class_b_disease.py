"""
CLASS B — Crop Disease & Protection Transformer

Scope:
- Disease descriptions
- Symptoms
- Affected crop stage
- Control methods (chemical / biological / IPM)

STRICTLY EXCLUDES:
- Registration numbers
- Manufacturer names
- Chemical composition
- Raw dosage tables

NO OCR
NO TABLES
NO EMBEDDINGS
"""

import re
import hashlib
from typing import List, Dict

from ingestion.transformers.base import (
    normalize_text,
    split_into_sections,
    enforce_length_limits,
    is_usable_page,
)

# =========================================================
# CONFIG
# =========================================================

MIN_SECTION_CHARS = 200
MAX_CHUNK_CHARS = 450

DROP_SECTION_KEYWORDS = [
    "registration",
    "chemical composition",
    "formulation",
    "dosage",
    "dose",
    "schedule",
    "toxicity",
    "manufacturer",
    "company",
    "license",
    "expiry",
]

ALLOWED_SECTION_HINTS = [
    "disease",
    "symptom",
    "affected",
    "stage",
    "management",
    "control",
    "prevention",
    "treatment",
    "ipm",
    "biological",
    "chemical",
]

# =========================================================
# HELPERS
# =========================================================

def _looks_like_dosage_text(text: str) -> bool:
    """
    Detect numeric-heavy dosage / chemical text.
    """
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    if digit_ratio > 0.35:
        return True

    if re.search(r"\b\d+(\.\d+)?\s*(ml|l|kg|g)/", text.lower()):
        return True

    return False


def _is_allowed_section(title: str) -> bool:
    t = title.lower()
    return any(h in t for h in ALLOWED_SECTION_HINTS)


def _is_dropped_section(title: str) -> bool:
    t = title.lower()
    return any(k in t for k in DROP_SECTION_KEYWORDS)


def _stable_chunk_id(doc_id: str, page: int, title: str, index: int) -> str:
    raw = f"{doc_id}|{page}|{title}|{index}"
    return hashlib.md5(raw.encode()).hexdigest()


# =========================================================
# MAIN TRANSFORMER
# =========================================================

def transform_class_b(
    pages: List[Dict],
    doc_rules: Dict,
) -> List[Dict]:
    """
    Disease transformer.

    Guarantees:
    - One disease aspect per chunk
    - No dosage leakage
    - Conservative confidence
    """

    domain = doc_rules.get("domain", "crop_disease")
    confidence_cap = min(doc_rules.get("confidence_cap", 0.8), 0.8)

    final_chunks: List[Dict] = []

    for page in pages:
        if not is_usable_page(page):
            continue

        page_number = page.get("page_number")
        source = page.get("source", "")
        doc_id = hashlib.md5(source.encode()).hexdigest()

        raw_text = normalize_text(page.get("text", ""))
        sections = split_into_sections(raw_text)

        for idx, sec in enumerate(sections):
            title = sec["title"]
            content = sec["content"]

            if len(content) < MIN_SECTION_CHARS:
                continue

            if _is_dropped_section(title):
                continue

            if not _is_allowed_section(title):
                continue

            if _looks_like_dosage_text(content):
                continue

            content = enforce_length_limits(content)[:MAX_CHUNK_CHARS]

            chunk_id = _stable_chunk_id(
                doc_id=doc_id,
                page=page_number,
                title=title,
                index=idx,
            )

            final_chunks.append({
                "text": content,
                "metadata": {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "page": page_number,
                    "source": source,
                    "domain": domain,
                    "section": title.lower(),
                    "content_type": "disease_info",
                    "confidence": confidence_cap,
                    "priority": 4,
                }
            })

    return final_chunks
