# ingestion/chunker.py

from typing import List, Dict
import re
import hashlib


# ---------- CONFIG (soft constraints, NOT goals) ----------
MAX_TOKENS_SOFT = 300        # approximate, not strict
MIN_CHARS = 40               # ignore garbage
OCR_MAX_CHARS = 180          # OCR chunks must be smaller


# ---------- UTILS ----------
def _approx_token_count(text: str) -> int:
    # rough heuristic: 1 token ≈ 4 chars
    return len(text) // 4


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _stable_chunk_id(doc_id: str, page: int, section: str, index: int) -> str:
    raw = f"{doc_id}_{page}_{section}_{index}"
    return hashlib.md5(raw.encode()).hexdigest()


# ---------- SEMANTIC SPLITTERS ----------
LOGICAL_SPLITS = [
    r"\bhowever\b",
    r"\bexcept\b",
    r"\bprovided that\b",
    r"\bnote:\b",
    r"\bimportant:\b",
]


def _split_logical_units(text: str) -> List[str]:
    """
    Split text by logical pivots, not size.
    """
    units = [text]
    for pattern in LOGICAL_SPLITS:
        temp = []
        for u in units:
            temp.extend(re.split(f"({pattern})", u, flags=re.IGNORECASE))
        units = temp

    merged = []
    buffer = ""
    for u in units:
        if re.match("|".join(LOGICAL_SPLITS), u, flags=re.IGNORECASE):
            buffer += " " + u
        else:
            if buffer:
                merged.append(buffer.strip())
            buffer = u

    if buffer.strip():
        merged.append(buffer.strip())

    return [u for u in merged if len(u.strip()) >= MIN_CHARS]


def _split_procedure_steps(text: str) -> List[str]:
    """
    One step = one chunk.
    """
    steps = re.split(r"\n\s*\d+[\).\s]", text)
    return [s.strip() for s in steps if len(s.strip()) >= MIN_CHARS]


def _split_paragraphs(text: str) -> List[str]:
    paras = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paras if len(p.strip()) >= MIN_CHARS]


# ---------- MAIN CHUNKER ----------
def chunk_page(page: Dict) -> List[Dict]:
    """
    Meaning-first chunker.
    Produces atomic, metadata-rich chunks.
    """

    text = _normalize_text(page["content"])
    content_type = page["content_type"]
    confidence = page.get("confidence", 1.0)

    doc_id = page["doc_id"]
    page_no = page["page_number"]
    section = page.get("section", "unknown")

    chunks = []
    chunk_index = 0

    # ---------- PROCEDURES ----------
    if content_type == "procedure":
        units = _split_procedure_steps(text)

    # ---------- OCR ----------
    elif content_type == "ocr":
        units = _split_paragraphs(text)
        units = [u for u in units if len(u) <= OCR_MAX_CHARS]

    # ---------- NORMAL TEXT ----------
    else:
        paras = _split_paragraphs(text)
        units = []
        for p in paras:
            units.extend(_split_logical_units(p))

    # ---------- BUILD CHUNKS ----------
    for unit in units:
        token_estimate = _approx_token_count(unit)

        if token_estimate > MAX_TOKENS_SOFT:
            # fallback: sentence split
            sentences = re.split(r"(?<=[.!?])\s+", unit)
            for s in sentences:
                if len(s.strip()) < MIN_CHARS:
                    continue
                chunks.append(_build_chunk(
                    s, page, section, chunk_index
                ))
                chunk_index += 1
            continue

        chunks.append(_build_chunk(
            unit, page, section, chunk_index
        ))
        chunk_index += 1

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
        "text": text,
        "metadata": {
            "chunk_id": chunk_id,
            "doc_id": page["doc_id"],
            "source": page["source"],
            "page_number": page["page_number"],
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
        "text": 4,
        "table": 5,
        "ocr": 2
    }.get(content_type, 3)
