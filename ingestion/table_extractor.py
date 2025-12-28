# ingestion/table_extractor.py

import camelot
import hashlib
from typing import List, Dict

# ---------------- CONFIG ----------------

MIN_ROWS = 2
MIN_COLS = 2
MAX_ROWS_PER_TABLE = 50          # 🔑 prevents chunk explosion
MAX_TEXT_LENGTH = 300            # 🔑 avoids prose tables
NUMERIC_HEAVY_RATIO = 0.6        # 🔑 numeric-dominant detection


# ---------------- HELPERS ----------------

def _stable_chunk_id(
    doc_id: str,
    page_number: int,
    table_index: int,
    row_index: int
) -> str:
    raw = f"{doc_id}_p{page_number}_t{table_index}_r{row_index}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _clean_cell(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _numeric_ratio(values: List[str]) -> float:
    if not values:
        return 0.0
    numeric = sum(
        1 for v in values
        if any(ch.isdigit() for ch in v)
    )
    return numeric / len(values)


# ---------------- MAIN EXTRACTOR ----------------

def extract_tables_from_pdf(
    pdf_path: str,
    page_number: int,
    page_meta: Dict
) -> List[Dict]:
    """
    Conservative, row-wise table extraction.

    Guarantees:
    - Table rows only (no prose tables)
    - Numeric-dominant bias
    - Stable chunk IDs
    - Hard caps to prevent noise
    """

    chunks: List[Dict] = []
    doc_id = page_meta["doc_id"]

    try:
        tables = camelot.read_pdf(
            pdf_path,
            pages=str(page_number),
            flavor="lattice",
            strip_text="\n",
        )
    except Exception:
        return chunks

    for table_index, table in enumerate(tables):

        df = table.df

        # ---------- BASIC STRUCTURE GUARD ----------
        if df.shape[0] < MIN_ROWS or df.shape[1] < MIN_COLS:
            continue

        headers = [_clean_cell(h) for h in df.iloc[0]]
        rows = df.iloc[1:].values.tolist()

        # ---------- HEADER QUALITY ----------
        if not any(headers):
            continue

        # ---------- ROW CAP ----------
        rows = rows[:MAX_ROWS_PER_TABLE]

        for row_index, row in enumerate(rows):

            if len(row) != len(headers):
                continue

            row_values = [_clean_cell(v) for v in row]

            # ---------- EMPTY / JUNK ROW ----------
            if not any(row_values):
                continue

            # ---------- NUMERIC DOMINANCE ----------
            if _numeric_ratio(row_values) < NUMERIC_HEAVY_RATIO:
                continue

            # ---------- BUILD MINIMAL TEXT ----------
            parts = []
            for h, v in zip(headers, row_values):
                if h and v:
                    parts.append(f"{h}: {v}")

            if not parts:
                continue

            chunk_text = " | ".join(parts)

            # ---------- PROSE TABLE GUARD ----------
            if len(chunk_text) > MAX_TEXT_LENGTH:
                continue

            chunk_id = _stable_chunk_id(
                doc_id=doc_id,
                page_number=page_number,
                table_index=table_index,
                row_index=row_index,
            )

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "page": page_number,
                    "source": page_meta.get("source"),
                    "content_type": "table_row",
                    "table_index": table_index,
                    "row_index": row_index,

                    # column-aware (critical for retriever intent checks)
                    "table_headers": headers,
                    "row_raw": row_values,

                    # provenance
                    "extraction_method": "camelot",
                    "source_type": page_meta.get("source_type", "pdf"),
                    "language": page_meta.get("language", "unknown"),
                    "confidence": min(page_meta.get("confidence", 1.0), 0.85),  # 🔑 table ≠ perfect
                    "priority": 5,
                }
            })

    return chunks
