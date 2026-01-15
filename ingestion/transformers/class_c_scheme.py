"""
CLASS C — Government Schemes / Guidelines Transformer

Purpose:
- Extract ONLY actionable scheme information
- Keep sections isolated (eligibility ≠ benefits ≠ process)
- Prevent policy hallucination

STRICTLY EXCLUDES:
- Preamble
- Ministry speeches
- Legal clauses
- Budget tables
- Historical background
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
# CONFIG — DO NOT RELAX
# =========================================================

MIN_SECTION_CHARS = 150
MAX_CHUNK_CHARS = 400

ALLOWED_SECTION_HINTS = [
    "eligibility",
    "benefit",
    "coverage",
    "premium",
    "subsidy",
    "assistance",
    "exclusion",
    "how to apply",
    "application",
    "procedure",
    "claim",
]

DROP_SECTION_KEYWORDS = [
    "preamble",
    "objective",
    "introduction",
    "background",
    "ministry",
    "department",
    "notification",
    "act",
    "clause",
    "section",
    "budget",
    "allocation",
    "financial outlay",
    "annexure",
    "appendix",
    "schedule",
    "form",
    "format",
]

# =========================================================
# HELPERS
# =========================================================

def _is_allowed_section(title: str) -> bool:
    t = title.lower()
    return any(h in t for h in ALLOWED_SECTION_HINTS)


def _is_dropped_section(title: str) -> bool:
    t = title.lower()
    return any(k in t for k in DROP_SECTION_KEYWORDS)


def _looks_like_legal_text(text: str) -> bool:
    return bool(
        re.search(r"\bshall\b|\bhereby\b|\bthereof\b|\bwhereas\b", text.lower())
    )


def _stable_chunk_id(doc_id: str, page: int, title: str, index: int) -> str:
    raw = f"{doc_id}|{page}|{title}|{index}"
    return hashlib.md5(raw.encode()).hexdigest()


def _infer_scheme_name(source: str, rules: Dict) -> str:
    if "scheme" in rules:
        return rules["scheme"]
    return source.split("/")[-1].replace(".pdf", "").lower()


# =========================================================
# MAIN TRANSFORMER
# =========================================================

def transform_class_c(
    pages: List[Dict],
    doc_rules: Dict,
) -> List[Dict]:
    """
    Transform scheme PDFs into safe, section-isolated chunks.

    Guarantees:
    - One scheme section per chunk
    - No eligibility/benefit mixing
    - No legal language leakage
    """

    domain = doc_rules.get("domain", "scheme")
    confidence_cap = min(doc_rules.get("confidence_cap", 0.9), 0.9)

    final_chunks: List[Dict] = []

    for page in pages:
        if not is_usable_page(page):
            continue

        page_number = page.get("page_number")
        source = page.get("source", "")
        doc_id = hashlib.md5(source.encode()).hexdigest()
        scheme_name = _infer_scheme_name(source, doc_rules)

        text = normalize_text(page.get("text", ""))
        sections = split_into_sections(text)

        for idx, sec in enumerate(sections):
            title = sec["title"]
            content = sec["content"]

            if len(content) < MIN_SECTION_CHARS:
                continue

            if _is_dropped_section(title):
                continue

            if not _is_allowed_section(title):
                continue

            if _looks_like_legal_text(content):
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
                    "scheme": scheme_name,
                    "chunk_type": title.lower(),
                    "content_type": "scheme_info",
                    "confidence": confidence_cap,
                    "priority": 4,
                }
            })

    return final_chunks
