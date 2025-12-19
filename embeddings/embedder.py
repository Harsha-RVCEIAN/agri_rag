from typing import List, Dict
import torch
from sentence_transformers import SentenceTransformer


# ---------------- CONFIG ----------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_VERSION = "minilm_v1"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- EMBEDDER ----------------

class Embedder:
    """
    Responsible ONLY for:
    - loading embedding model
    - embedding text (chunks or queries)
    - normalizing vectors
    """

    def __init__(self):
        self.model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device=DEVICE
        )
        self.model.eval()

    # -------------------------------------------------
    # INTERNAL CORE EMBEDDING
    # -------------------------------------------------

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        vectors: List[List[float]] = []

        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]

            with torch.no_grad():
                embs = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )

            vectors.extend(embs.tolist())

        return vectors

    # -------------------------------------------------
    # QUERY EMBEDDING (NEW — REQUIRED)
    # -------------------------------------------------

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed raw query texts.
        Returns list of vectors.
        """
        clean_texts = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
        return self._embed(clean_texts)

    # -------------------------------------------------
    # CHUNK EMBEDDING (EXISTING)
    # -------------------------------------------------

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Embed document chunks for vector store upsert.
        """

        valid_chunks = [
            c for c in chunks
            if isinstance(c.get("text"), str)
            and c["text"].strip()
            and isinstance(c.get("metadata"), dict)
            and c["metadata"].get("chunk_id")
        ]

        if not valid_chunks:
            return []

        texts = [c["text"] for c in valid_chunks]
        vectors = self._embed(texts)

        records: List[Dict] = []

        for chunk, vector in zip(valid_chunks, vectors):
            meta = chunk["metadata"]

            records.append({
                "id": meta["chunk_id"],
                "vector": vector,
                "metadata": {
                    **meta,
                    "text": chunk["text"],  # required for RAG
                    "embedding_version": EMBEDDING_VERSION,
                    "embedding_model": EMBEDDING_MODEL_NAME
                }
            })

        return records
