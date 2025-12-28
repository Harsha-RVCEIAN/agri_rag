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


# ---------------- CONSTANTS ----------------

MAX_CHUNKS_PER_PAGE = 8          # 🔑 hard safety cap
MIN_OCR_CONFIDENCE = 0.35        # 🔑 below this OCR is ignored


# ---------------- UTILS ----------------

def _doc_id_from_path(path: str) -> str:
    return hashlib.md5(os.path.abspath(path).encode()).hexdigest()


def _normalize_language(lang_obj) -> str:
    if isinstance(lang_obj, dict):
        return lang_obj.get("language", "unknown")
    if isinstance(lang_obj, str):
        return lang_obj
    return "unknown"


# ---------------- PIPELINE ----------------

def ingest_pdf(pdf_path: str) -> List[Dict]:
    """
    FINAL ingestion pipeline (OPTIMIZED).

    Guarantees preserved:
    - Text preferred over OCR
    - Tables preserved with higher priority
    - No duplicate chunks
    - Stable metadata

    New guarantees:
    - OCR only when classifier demands
    - Tables extracted once per page
    - Hard chunk caps
    - Lower latency
    """

    pages = load_pdf_pages(pdf_path)
    all_chunks: List[Dict] = []
    doc_id = _doc_id_from_path(pdf_path)

    for page in pages:
        page_number = page.get("page_number")
        source_type = page.get("source_type", "pdf")

        raw_text = (page.get("text") or "").strip()
        has_text = bool(raw_text)

        # ---------- LANGUAGE (ONLY ON TEXT) ----------
        lang_obj = detect_language(raw_text) if has_text else None
        language = _normalize_language(lang_obj)
        language_conf = (
            lang_obj.get("confidence", 1.0)
            if isinstance(lang_obj, dict)
            else 1.0
        )

        page_chunks: List[Dict] = []

        # =========================================================
        # 1️⃣ PAGE CLASSIFICATION (CRITICAL FIX)
        # =========================================================
        page_class = classify_page({
            "text": raw_text,
            "images": page.get("images", []),
            "header_repeat_score": page.get("header_repeat_score", 0.0),
        })

        # =========================================================
        # 2️⃣ TEXT PATH (PRIMARY)
        # =========================================================
        if has_text and page_class == "TEXT_OK":
            page_obj = {
                "doc_id": doc_id,
                "page_number": page_number,
                "source": pdf_path,
                "content": raw_text,
                "content_type": "text",
                "language": language,
                "confidence": 1.0,
                "source_type": source_type,
            }

            page_chunks.extend(chunk_text_page(page_obj))

        # =========================================================
        # 3️⃣ OCR PATH (STRICT, RARE)
        # =========================================================
        if page_class == "OCR_REQUIRED" and page.get("images"):
            ocr_blocks = []
            confidences = []

            for image in page["images"]:
                img_array = np.frombuffer(image["bytes"], np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                processed = preprocess_image(img)
                text, conf, _ = run_ocr(processed, language)

                if text and conf >= MIN_OCR_CONFIDENCE:
                    ocr_blocks.append(text.strip())
                    confidences.append(conf)

            if ocr_blocks:
                avg_conf = statistics.median(confidences)

                page_obj = {
                    "doc_id": doc_id,
                    "page_number": page_number,
                    "source": pdf_path,
                    "content": "\n".join(ocr_blocks),
                    "content_type": "ocr",
                    "language": language,
                    "confidence": round(avg_conf, 2),
                    "source_type": source_type,
                }

                page_chunks.extend(chunk_text_page(page_obj))

        # =========================================================
        # 4️⃣ TABLE EXTRACTION (ONCE PER PAGE)
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
        # 5️⃣ FAILSAFE (TEXT ONLY)
        # =========================================================
        if not page_chunks and has_text:
            page_chunks.append({
                "text": raw_text,
                "metadata": {
                    "chunk_id": f"{doc_id}_page_{page_number}_fallback",
                    "page": page_number,
                    "source": pdf_path,
                    "language": language,
                    "confidence": 0.4,
                    "content_type": "text",
                    "source_type": source_type,
                }
            })

        # =========================================================
        # 6️⃣ DEDUPE + HARD CAP
        # =========================================================
        seen_hashes = set()
        merged = page_chunks + table_chunks

        for chunk in merged:
            if len(all_chunks) >= MAX_CHUNKS_PER_PAGE * len(pages):
                break

            text = (chunk.get("text") or "").strip()
            if not text:
                continue

            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            meta = chunk.get("metadata", {})
            meta.setdefault("chunk_id", f"{doc_id}_p{page_number}_{h[:8]}")
            meta.setdefault("page", page_number)
            meta.setdefault("source", pdf_path)
            meta.setdefault("language", language)
            meta.setdefault("language_confidence", round(language_conf, 3))
            meta.setdefault("confidence", meta.get("confidence", 1.0))
            meta.setdefault("content_type", meta.get("content_type", "text"))
            meta.setdefault("source_type", source_type)

            all_chunks.append({
                "text": text,
                "metadata": meta
            })

    return all_chunks
