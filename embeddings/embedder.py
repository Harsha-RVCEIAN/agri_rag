# embeddings/embedder.py

from typing import List, Dict
import torch
from sentence_transformers import SentenceTransformer
import numpy as np


# ---------------- CONFIG ----------------

EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_VERSION = "bge_v1"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- EMBEDDER ----------------

class Embedder:
    """
    Responsible ONLY for:
    - loading embedding model
    - embedding clean chunk text
    - normalizing vectors
    - returning ready-to-upsert records
    """

    def __init__(self):
        self.model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device=DEVICE
        )
        self.model.eval()

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Embed a list of chunks.

        Input chunk format:
        {
            "text": str,
            "metadata": {...}
        }

        Output record format:
        {
            "id": chunk_id,
            "vector": List[float],
            "metadata": {...}
        }
        """

        records: List[Dict] = []

        # ---- sanity filter ----
        valid_chunks = [
            c for c in chunks
            if c.get("text") and c.get("metadata", {}).get("chunk_id")
        ]

        if not valid_chunks:
            return records

        texts = [c["text"] for c in valid_chunks]

        # ---- batch embedding ----
        for start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[start:start + BATCH_SIZE]

            with torch.no_grad():
                embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,  # CRITICAL
                    show_progress_bar=False
                )

            for i, vector in enumerate(embeddings):
                chunk = valid_chunks[start + i]
                meta = chunk["metadata"]

                records.append({
                    "id": meta["chunk_id"],           # deterministic ID
                    "vector": vector.tolist(),        # normalized
                    "metadata": {
                        **meta,
                        "embedding_version": EMBEDDING_VERSION,
                        "embedding_model": EMBEDDING_MODEL_NAME
                    }
                })

        return records
