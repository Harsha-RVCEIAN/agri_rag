# ingestion/table_extractor.py

import camelot
import hashlib
from typing import List, Dict


# ---------------- HELPERS ----------------

def _stable_chunk_id(
    doc_id: str,
    page_number: int,
    table_index: int,
    row_index: int
) -> str:
    """
    Deterministic chunk ID for table rows.
    """
    raw = f"{doc_id}_p{page_number}_t{table_index}_r{row_index}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _clean_cell(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


# ---------------- MAIN EXTRACTOR ----------------

def extract_tables_from_pdf(
    pdf_path: str,
    page_number: int,
    page_meta: Dict
) -> List[Dict]:
    """
    Extract tables from a given page using Camelot.
    Returns row-wise, retrieval-safe table chunks.
    """

    chunks: List[Dict] = []

    doc_id = page_meta["doc_id"]

    try:
        tables = camelot.read_pdf(
            pdf_path,
            pages=str(page_number),
            flavor="lattice",   # best for govt PDFs
            strip_text="\n"
        )
    except Exception:
        return chunks

    for table_index, table in enumerate(tables):
        df = table.df

        # ---- skip junk tables ----
        if df.shape[0] < 2 or df.shape[1] < 2:
            continue

        headers = [_clean_cell(h) for h in df.iloc[0]]
        rows = df.iloc[1:].values.tolist()

        for row_index, row in enumerate(rows):
            if len(row) != len(headers):
                continue

            row_values = [_clean_cell(v) for v in row]

            # ---- minimal embedding text (not prose) ----
            text_parts = []
            for h, v in zip(headers, row_values):
                if h and v:
                    text_parts.append(f"{h}: {v}")

            if not text_parts:
                continue

            chunk_text = " | ".join(text_parts)

            chunk_id = _stable_chunk_id(
                doc_id=doc_id,
                page_number=page_number,
                table_index=table_index,
                row_index=row_index
            )

            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "page_number": page_number,
                    "content_type": "table_row",
                    "table_index": table_index,
                    "row_index": row_index,

                    # ---- column-aware metadata (CRITICAL) ----
                    "table_headers": headers,
                    "row_raw": row_values,

                    # ---- provenance ----
                    "extraction_method": "camelot",
                    "source_type": page_meta.get("source_type", "pdf"),
                    "language": page_meta.get("language", "unknown"),
                    "confidence": page_meta.get("confidence", 1.0),
                    "priority": 5
                }
            })

    return chunks
