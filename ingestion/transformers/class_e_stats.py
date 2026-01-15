"""
CLASS E — Statistics / Annual / Research Reports Transformer

Purpose:
- Convert massive reports into reference-only summaries
- Prevent tables, annexures, appendices from entering vector DB
- Ensure these documents NEVER dominate answers

STRICT RULES:
- NO page-wise chunks
- NO tables
- NO numeric-heavy sections
- ONLY high-level summaries
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
# CONFIG — DO NOT WEAKEN
# =========================================================

MIN_SECTION_CHARS = 250
MAX_SUMMARY_CHARS = 350
MAX_TOTAL_CHUNKS = 30          # 🔑 HARD CAP PER DOCUMENT

ALLOWED_SECTION_HINTS = [
    "overview",
    "summary",
    "key findings",
    "highlights",
    "introduction",
    "conclusion",
    "discussion",
    "analysis",
]

DROP_SECTION_KEYWORDS = [
    "table",
    "annexure",
    "appendix",
    "schedule",
    "statement",
    "data",
    "figure",
    "chart",
    "graph",
    "methodology",
    "method",
    "survey design",
    "sample",
    "questionnaire",
    "references",
    "bibliography",
    "index",
]

NUMERIC_HEAVY_RATIO = 0.30

# =========================================================
# HELPERS
# =========================================================

def _is_allowed_section(title: str) -> bool:
    t = title.lower()
    return any(h in t for h in ALLOWED_SECTION_HINTS)


def _is_dropped_section(title: str) -> bool:
    t = title.lower()
    return any(k in t for k in DROP_SECTION_KEYWORDS)


def _numeric_ratio(text: str) -> float:
    digits = sum(c.isdigit() for c in text)
    return digits / max(len(text), 1)


def _looks_like_statistical_dump(text: str) -> bool:
    if _numeric_ratio(text) > NUMERIC_HEAVY_RATIO:
        return True
    if re.search(r"\b\d{4}\b", text):   # excessive year mentions
        return True
    return False


def _stable_chunk_id(doc_id: str, page: int, title: str, index: int) -> str:
    raw = f"{doc_id}|{page}|{title}|{index}"
    return hashlib.md5(raw.encode()).hexdigest()


# =========================================================
# MAIN TRANSFORMER
# =========================================================

def transform_class_e(
    pages: List[Dict],
    doc_rules: Dict,
) -> List[Dict]:
    """
    Transform large reports into safe, reference-only summaries.

    Guarantees:
    - Low-priority, background-only usage
    - No numeric or tabular dominance
    - Hard chunk cap
    """

    domain = doc_rules.get("domain", "statistics")
    confidence_cap = min(doc_rules.get("confidence_cap", 0.5), 0.5)

    final_chunks: List[Dict] = []

    for page in pages:
        if len(final_chunks) >= MAX_TOTAL_CHUNKS:
            break

        if not is_usable_page(page):
            continue

        page_number = page.get("page_number")
        source = page.get("source", "")
        doc_id = hashlib.md5(source.encode()).hexdigest()

        text = normalize_text(page.get("text", ""))
        sections = split_into_sections(text)

        for idx, sec in enumerate(sections):
            if len(final_chunks) >= MAX_TOTAL_CHUNKS:
                break

            title = sec["title"]
            content = sec["content"]

            if len(content) < MIN_SECTION_CHARS:
                continue

            if _is_dropped_section(title):
                continue

            if not _is_allowed_section(title):
                continue

            if _looks_like_statistical_dump(content):
                continue

            content = enforce_length_limits(content)[:MAX_SUMMARY_CHARS]

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
                    "content_type": "reference_summary",
                    "use": "reference_only",
                    "confidence": confidence_cap,
                    "priority": 1,   # 🔑 lowest priority
                }
            })

    return final_chunks
