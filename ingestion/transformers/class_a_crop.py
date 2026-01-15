"""
CLASS A — Crop Production Transformer

Purpose:
- Clean cultivation PDFs
- Remove junk sections
- Produce farmer-relevant chunks
- Enrich metadata (crop, chunk_type)

This file MUST NOT:
- Do OCR
- Handle tables
- Call embeddings
"""

import re
import hashlib
from typing import List, Dict, Optional

from ingestion.transformers.base import (
    normalize_text,
    is_usable_page,
    split_into_sections,
    enforce_length_limits,
)

# =========================================================
# CONFIG
# =========================================================

SECTION_MAP = {
    "sowing": "schedule",
    "time of sowing": "schedule",
    "season": "schedule",
    "soil": "requirement",
    "climate": "requirement",
    "fertilizer": "practice",
    "manure": "practice",
    "nutrient": "practice",
    "irrigation": "practice",
    "water": "practice",
    "varieties": "practice",
    "seed rate": "practice",
    "spacing": "practice",
    "harvesting": "practice",
    "yield": "practice",
}

MIN_SECTION_CHARS = 250
MAX_SECTION_CHARS = 3000


# =========================================================
# HELPERS
# =========================================================

def _infer_chunk_type(title: str) -> str:
    t = title.lower()
    for k, v in SECTION_MAP.items():
        if k in t:
            return v
    return "practice"


def _infer_crop_from_filename(filename: str) -> Optional[str]:
    name = filename.lower()
    crops = [
        "rice", "wheat", "cotton", "maize", "millet",
        "sorghum", "barley", "paddy", "ragi",
    ]
    for c in crops:
        if c in name:
            return c
    return None


def _stable_chunk_id(doc_id: str, page: int, title: str, index: int) -> str:
    raw = f"{doc_id}|{page}|{title}|{index}"
    return hashlib.md5(raw.encode()).hexdigest()


# =========================================================
# CORE TRANSFORMER
# =========================================================

def transform_class_a(
    pages: List[Dict],
    pdf_path: str,
    doc_rules: Dict
) -> List[Dict]:
    """
    Transform raw pages into Class-A (crop production) chunks.
    """

    filename = pdf_path.split("/")[-1]
    crop = _infer_crop_from_filename(filename)
    doc_id = hashlib.md5(pdf_path.encode()).hexdigest()

    confidence_cap = doc_rules.get("confidence_cap", 1.0)

    final_chunks: List[Dict] = []

    # =====================================================
    # PAGE LOOP (STRICT)
    # =====================================================

    for page in pages:
        if not is_usable_page(page):
            continue

        page_number = page.get("page_number")
        raw_text = normalize_text(page.get("text", ""))

        sections = split_into_sections(raw_text)

        for idx, sec in enumerate(sections):
            title = sec["title"]
            content = sec["content"]

            if len(content) < MIN_SECTION_CHARS:
                continue

            content = enforce_length_limits(content)

            chunk_type = _infer_chunk_type(title)

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
                    "source": pdf_path,
                    "domain": "crop_production",
                    "crop": crop,
                    "chunk_type": chunk_type,
                    "section": title,
                    "content_type": "text",
                    "confidence": confidence_cap,
                    "priority": 4,
                }
            })

    return final_chunks
