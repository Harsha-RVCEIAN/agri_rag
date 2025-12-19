# ingestion/chunker.py

from typing import List, Dict
import re
import hashlib


# ---------- CONFIG ----------
MAX_TOKENS_SOFT = 300
MIN_CHARS = 40
OCR_MAX_CHARS = 220   # slightly higher to preserve meaning


# ---------- UTILS ----------
def _approx_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _stable_chunk_id(doc_id: str, page: int, section: str, index: int) -> str:
    raw = f"{doc_id}|{page}|{section}|{index}"
    return hashlib.md5(raw.encode()).hexdigest()


# ---------- SEMANTIC SPLITTERS ----------
LOGICAL_PIVOTS = [
    "however",
    "except",
    "provided that",
    "note:",
    "important:",
]


def _split_logical_units(text: str) -> List[str]:
    """
    Split text so that logical pivots START new chunks.
    """
    pattern = r"(?i)\b(" + "|".join(LOGICAL_PIVOTS) + r")\b"
    parts = re.split(pattern, text)

    units = []
    buffer = ""

    for part in parts:
        if part.lower() in LOGICAL_PIVOTS:
            if buffer.strip():
                units.append(buffer.strip())
            buffer = part
        else:
            buffer += " " + part

    if buffer.strip():
        units.append(buffer.strip())

    return [u for u in units if len(u) >= MIN_CHARS]


def _split_procedure_steps(text: str) -> List[str]:
    steps = re.split(r"\n?\s*\d+[\).\s]", text)
    return [s.strip() for s in steps if len(s.strip()) >= MIN_CHARS]


def _split_paragraphs(text: str) -> List[str]:
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if len(p.strip()) >= MIN_CHARS]


# ---------- MAIN CHUNKER ----------
def chunk_page(page: Dict) -> List[Dict]:
    """
    Meaning-first chunker.
    Produces stable, relevance-friendly chunks.
    """

    text = _normalize_text(page["content"])
    content_type = page["content_type"]

    doc_id = page["doc_id"]
    page_no = page["page_number"]

    # Better section inference
    section = page.get("section") or content_type

    units: List[str] = []

    # ---------- PROCEDURE ----------
    if content_type == "procedure":
        units = _split_procedure_steps(text)

    # ---------- OCR ----------
    elif content_type == "ocr":
        paras = _split_paragraphs(text)
        units = []
        for p in paras:
            if len(p) <= OCR_MAX_CHARS:
                units.append(p)
            else:
                units.extend(
                    re.split(r"(?<=[.!?])\s+", p)
                )

    # ---------- NORMAL TEXT ----------
    else:
        paras = _split_paragraphs(text)
        for p in paras:
            units.extend(_split_logical_units(p))

    # ---------- BUILD CHUNKS ----------
    chunks: List[Dict] = []
    index = 0

    for unit in units:
        if _approx_token_count(unit) > MAX_TOKENS_SOFT:
            sentences = re.split(r"(?<=[.!?])\s+", unit)
            buffer = ""
            for s in sentences:
                buffer += " " + s
                if _approx_token_count(buffer) >= MAX_TOKENS_SOFT:
                    chunks.append(_build_chunk(buffer.strip(), page, section, index))
                    index += 1
                    buffer = ""
            if buffer.strip():
                chunks.append(_build_chunk(buffer.strip(), page, section, index))
                index += 1
        else:
            chunks.append(_build_chunk(unit, page, section, index))
            index += 1

    return chunks


# ---------- CHUNK BUILDER ----------
def _build_chunk(text: str, page: Dict, section: str, index: int) -> Dict:
    chunk_id = _stable_chunk_id(
        page["doc_id"],
        page["page_number"],
        section,
        index
    )

    return {
        "text": text.strip(),
        "metadata": {
            "chunk_id": chunk_id,
            "doc_id": page["doc_id"],
            "source": page["source"],
            "page": page["page_number"],
            "section": section,
            "content_type": page["content_type"],
            "language": page["language"],
            "confidence": page.get("confidence", 1.0),
            "priority": _priority_from_type(page["content_type"])
        }
    }


def _priority_from_type(content_type: str) -> int:
    return {
        "procedure": 5,
        "table": 5,
        "table_row": 5,
        "text": 4,
        "ocr": 2
    }.get(content_type, 3)
