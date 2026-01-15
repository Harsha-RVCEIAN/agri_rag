"""
Ingestion package.

Responsible for document loading, OCR, chunking,
and preprocessing before embedding.
"""

from ingestion.pipeline import ingest_pdf

__all__ = [
    "ingest_pdf",
]
