from typing import List, Dict
import torch
from sentence_transformers import SentenceTransformer
from functools import lru_cache


# ---------------- CONFIG ----------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_VERSION = "minilm_v1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64 if DEVICE == "cuda" else 16


# ---------------- GLOBAL SINGLETON ----------------

_MODEL = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device=DEVICE
        )
        _MODEL.eval()
    return _MODEL


# ---------------- EMBEDDER ----------------

class Embedder:
    """
    Responsible ONLY for:
    - embedding queries
    - embedding document chunks
    - returning normalized vectors
    """

    def __init__(self):
        self.model = _get_model()

    # -------------------------------------------------
    # CORE EMBEDDING
    # -------------------------------------------------

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        vectors: List[List[float]] = []

        with torch.no_grad():
            for start in range(0, len(texts), BATCH_SIZE):
                batch = texts[start:start + BATCH_SIZE]

                embs = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                vectors.extend(embs.tolist())

        return vectors

    # -------------------------------------------------
    # QUERY EMBEDDING (CACHED)
    # -------------------------------------------------

    @lru_cache(maxsize=1024)
    def _embed_single_query(self, text: str) -> List[float]:
        """
        Cached single-query embedding.
        """
        vecs = self._embed([text])
        return vecs[0] if vecs else []

    def embed_query(self, query: str) -> List[float]:
        """
        PUBLIC API for query embedding.
        This is what Retriever / Pipeline should call.
        """
        if not isinstance(query, str) or not query.strip():
            return []
        return self._embed_single_query(query.strip())

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts.
        """
        clean = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
        if not clean:
            return []

        if len(clean) == 1:
            return [self._embed_single_query(clean[0])]

        return self._embed(clean)

    # -------------------------------------------------
    # CHUNK EMBEDDING
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
                    "text": chunk["text"],
                    "embedding_version": EMBEDDING_VERSION,
                    "embedding_model": EMBEDDING_MODEL_NAME,
                }
            })

        return records
