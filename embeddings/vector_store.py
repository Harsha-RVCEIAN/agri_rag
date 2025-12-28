import os
import time
from typing import List, Dict, Optional
from functools import lru_cache

from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone, ServerlessSpec


# ---------------- CONFIG ----------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agri-rag")

EMBEDDING_DIM = 384
METRIC = "cosine"

MIN_RAW_SCORE = 0.22
UPSERT_BATCH_SIZE = 100

DEFAULT_NAMESPACE = "agriculture"

REQUIRED_METADATA_FIELDS = {
    "chunk_id",
    "content_type",
    "source",
}


# ---------------- GLOBAL SINGLETON ----------------
# 🔑 Avoid repeated Pinecone init & index checks

_PC = None
_INDEX = None


def _get_index():
    global _PC, _INDEX

    if _INDEX is not None:
        return _INDEX

    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY not set")

    if _PC is None:
        _PC = Pinecone(api_key=PINECONE_API_KEY)

    existing = {i["name"] for i in _PC.list_indexes()}

    if INDEX_NAME not in existing:
        _PC.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric=METRIC,
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_ENV
            )
        )

        # non-busy wait (polite)
        while True:
            status = _PC.describe_index(INDEX_NAME).status
            if status.get("ready"):
                break
            time.sleep(0.5)

    _INDEX = _PC.Index(INDEX_NAME)
    return _INDEX


# ---------------- VECTOR STORE ----------------

class VectorStore:
    """
    Pinecone-backed vector store.
    Optimized for:
    - low startup latency
    - minimal network roundtrips
    - strict safety guarantees
    """

    def __init__(self):
        self.index = _get_index()

    # -------------------------------------------------
    # UPSERT (BATCHED, SAFE)
    # -------------------------------------------------

    def upsert(
        self,
        records: List[Dict],
        namespace: str = DEFAULT_NAMESPACE
    ) -> None:

        if not records:
            return

        for i in range(0, len(records), UPSERT_BATCH_SIZE):
            batch = records[i:i + UPSERT_BATCH_SIZE]

            vectors = [
                (r["id"], r["vector"], r.get("metadata", {}))
                for r in batch
                if (
                    r.get("id")
                    and isinstance(r.get("vector"), list)
                    and len(r["vector"]) == EMBEDDING_DIM
                )
            ]

            if vectors:
                self.index.upsert(
                    vectors=vectors,
                    namespace=namespace
                )

    # -------------------------------------------------
    # QUERY (HOT PATH OPTIMIZED)
    # -------------------------------------------------

    def query(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None,
        namespace: str = DEFAULT_NAMESPACE
    ) -> List[Dict]:

        # ---- fast reject ----
        if not query_vector or len(query_vector) != EMBEDDING_DIM:
            return []

        try:
            response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filters,
                namespace=namespace
            )
        except Exception:
            return []

        matches = response.get("matches") or []
        if not matches:
            return []

        cleaned: List[Dict] = []

        for m in matches:
            score = m.get("score", 0.0)
            if score < MIN_RAW_SCORE:
                continue

            meta = m.get("metadata") or {}
            if not REQUIRED_METADATA_FIELDS.issubset(meta):
                continue

            cleaned.append({
                "id": m.get("id"),
                "score": score,
                "metadata": meta
            })

        return cleaned

    # -------------------------------------------------
    # DELETE / RESET
    # -------------------------------------------------

    def delete_by_doc(self, doc_id: str, namespace: str = DEFAULT_NAMESPACE) -> None:
        self.index.delete(
            filter={"doc_id": {"$eq": doc_id}},
            namespace=namespace
        )

    def delete_by_embedding_version(
        self,
        embedding_version: str,
        namespace: str = DEFAULT_NAMESPACE
    ) -> None:
        self.index.delete(
            filter={"embedding_version": {"$eq": embedding_version}},
            namespace=namespace
        )

    def reset(self, namespace: str = DEFAULT_NAMESPACE) -> None:
        self.index.delete(delete_all=True, namespace=namespace)
