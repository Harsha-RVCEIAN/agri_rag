# ingestion/pdf_loader.py

import pdfplumber
import fitz  # PyMuPDF
from typing import List, Dict
import hashlib
from collections import Counter


# ---------------- CONFIG ----------------

HEADER_FOOTER_LINES = 2   # top + bottom lines to consider


# ---------------- HELPERS ----------------

def _normalize_line(text: str) -> str:
    text = text.strip().lower()
    text = " ".join(text.split())
    return text


def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _extract_header_footer_candidates(text: str) -> List[str]:
    """
    Extract top and bottom lines as header/footer candidates.
    """
    if not text:
        return []

    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return []

    candidates = lines[:HEADER_FOOTER_LINES] + lines[-HEADER_FOOTER_LINES:]
    return [_normalize_line(c) for c in candidates if len(c) > 10]


# ---------------- MAIN LOADER ----------------

def load_pdf_pages(pdf_path: str) -> List[Dict]:
    """
    Load PDF pages and compute header/footer repeat scores.
    """

    pages: List[Dict] = []

    plumber_pdf = pdfplumber.open(pdf_path)
    fitz_pdf = fitz.open(pdf_path)

    header_footer_hashes: List[str] = []

    # ---------- FIRST PASS: EXTRACT RAW PAGES ----------
    for page_index in range(len(plumber_pdf.pages)):
        plumber_page = plumber_pdf.pages[page_index]
        fitz_page = fitz_pdf[page_index]

        # ---- TEXT EXTRACTION (PRIMARY) ----
        text = plumber_page.extract_text(
            x_tolerance=2,
            y_tolerance=2,
            layout=True
        )
        text = text.strip() if text else ""

        # ---- FALLBACK ----
        if not text or len(text) < 20:
            fallback_text = fitz_page.get_text("text")
            if fallback_text and len(fallback_text.strip()) > len(text):
                text = fallback_text.strip()

        # ---- IMAGE EXTRACTION ----
        images = []
        for img_index, img in enumerate(fitz_page.get_images(full=True)):
            xref = img[0]
            base_image = fitz_pdf.extract_image(xref)
            images.append({
                "image_index": img_index,
                "xref": xref,
                "bytes": base_image["image"],
                "ext": base_image["ext"],
                "width": base_image.get("width"),
                "height": base_image.get("height"),
            })

        # ---- HEADER/FOOTER CANDIDATES ----
        candidates = _extract_header_footer_candidates(text)
        for c in candidates:
            header_footer_hashes.append(_hash_text(c))

        pages.append({
            "page_number": page_index + 1,
            "text": text,
            "images": images,
            "width": fitz_page.rect.width,
            "height": fitz_page.rect.height,
            "rotation": fitz_page.rotation,
            "source_type": "pdf",
            "_header_footer_candidates": candidates,  # temp
        })

    # ---------- SECOND PASS: COMPUTE REPEAT SCORE ----------
    total_pages = len(pages)
    freq = Counter(header_footer_hashes)

    for page in pages:
        scores = []
        for c in page["_header_footer_candidates"]:
            h = _hash_text(c)
            scores.append(freq[h] / total_pages)

        page["header_repeat_score"] = max(scores) if scores else 0.0
        del page["_header_footer_candidates"]

    plumber_pdf.close()
    fitz_pdf.close()

    return pages
