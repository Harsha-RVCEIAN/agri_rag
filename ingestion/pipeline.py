from typing import List, Dict
import cv2
import numpy as np
import statistics
import os
import hashlib

from ingestion.pdf_loader import load_pdf_pages
from ingestion.page_classifier import classify_page
from ingestion.image_preprocess import preprocess_image
from ingestion.ocr_engine import run_ocr
from ingestion.language_router import route_language as detect_language
from ingestion.chunker import chunk_page as chunk_text_page
from ingestion.table_extractor import extract_tables_from_pdf


# ---------------- UTILS ----------------

def _doc_id_from_path(path: str) -> str:
    return hashlib.md5(os.path.abspath(path).encode()).hexdigest()


# ---------------- PIPELINE ----------------

def ingest_pdf(pdf_path: str) -> List[Dict]:
    """
    Robust ingestion pipeline.

    GUARANTEE:
    - If readable text exists → at least ONE chunk is produced.
    """

    pages = load_pdf_pages(pdf_path)
    all_chunks: List[Dict] = []

    doc_id = _doc_id_from_path(pdf_path)

    for page in pages:
        page_number = page.get("page_number")
        source_type = page.get("source_type", "pdf")

        raw_text = page.get("text", "") or ""
        has_text = bool(raw_text.strip())

        lang_obj = detect_language(raw_text) if has_text else None
        language = lang_obj["language"] if isinstance(lang_obj, dict) else "unknown"
        language_conf = lang_obj.get("confidence", 1.0) if isinstance(lang_obj, dict) else 1.0


        page_chunks: List[Dict] = []

        # =========================================================
        # TEXT PATH (PRIMARY)
        # =========================================================
        if has_text:
            page_obj = {
                "doc_id": doc_id,              # ✅ REQUIRED
                "page_number": page_number,
                "source": pdf_path, 
                "content": raw_text.strip(),
                "content_type": "text",
                "language": language,
                "confidence": 1.0,
                "source_type": source_type,
            }

            page_chunks.extend(chunk_text_page(page_obj))

        # =========================================================
        # OCR PATH (SECONDARY)
        # =========================================================
        if not page_chunks and page.get("images"):
            ocr_text_blocks = []
            confidences = []
            detected_languages = []

            for image in page["images"]:
                img_array = np.frombuffer(image["bytes"], np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is None:
                    continue

                processed_img = preprocess_image(img)
                text, conf, _ = run_ocr(processed_img, language)

                if text and text.strip():
                    ocr_text_blocks.append(text.strip())
                    confidences.append(conf)
                    detected_languages.append(detect_language(text))

            if ocr_text_blocks:
                avg_conf = statistics.median(confidences)

                if detected_languages:
                    language = max(
                        set(detected_languages),
                        key=detected_languages.count
                    )

                page_obj = {
                    "doc_id": doc_id,          # ✅ REQUIRED
                    "page_number": page_number,
                    "source": pdf_path, 
                    "content": "\n".join(ocr_text_blocks),
                    "content_type": "ocr",
                    "language": language,
                    "confidence": round(avg_conf, 2),
                    "source_type": source_type,
                }

                page_chunks.extend(chunk_text_page(page_obj))

        # =========================================================
        # TABLE EXTRACTION (NON-BLOCKING)
        # =========================================================
        try:
            table_chunks = extract_tables_from_pdf(
                pdf_path=pdf_path,
                page_number=page_number,
                page_meta={
                    "page_number": page_number,
                    "language": language,
                    "source_type": source_type,
                }
            )
        except Exception:
            table_chunks = []

        # =========================================================
        # PAGE-LEVEL FAILSAFE
        # =========================================================
        if not page_chunks and has_text:
            page_chunks.append({
                "text": raw_text.strip(),
                "metadata": {
                    "chunk_id": f"{doc_id}_page_{page_number}",
                    "page": page_number,
                    "source": pdf_path,
                    "language": language,
                    "confidence": 0.5,
                    "content_type": "fallback_text",
                    "source_type": source_type,
                }
            })

        # =========================================================
        # FINAL NORMALIZATION
        # =========================================================
        for chunk in page_chunks + table_chunks:
            text = chunk.get("text") or chunk.get("content", "")
            if not text or not text.strip():
                continue

            meta = chunk.get("metadata", {})

            meta.setdefault("chunk_id", f"{doc_id}_page_{page_number}")
            meta.setdefault("page", page_number)
            meta.setdefault("source", pdf_path)
            meta.setdefault("language", language)          # string only
            meta.setdefault("language_confidence", round(language_conf, 3))
            meta.setdefault("confidence", meta.get("confidence", 1.0))
            meta.setdefault("content_type", "text")
            meta.setdefault("source_type", source_type)

            all_chunks.append({
                "text": text.strip(),
                "metadata": meta
            })

    return all_chunks
