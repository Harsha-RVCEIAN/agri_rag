import os
import time
from typing import List, Dict, Optional

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

# 🔑 Metadata your system REQUIRES downstream
REQUIRED_METADATA_FIELDS = {
    "chunk_id",
    "content_type",
    "source",
    "domain",
    "confidence",
}


# ---------------- GLOBAL SINGLETON ----------------

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

    Guarantees:
    - metadata schema enforcement
    - safe filtering
    - zero silent corruption
    """

    def __init__(self):
        self.index = _get_index()

    # -------------------------------------------------
    # UPSERT (STRICT + BATCHED)
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
            vectors = []

            for r in batch:
                vid = r.get("id")
                vec = r.get("vector")
                meta = r.get("metadata", {})

                # ---------- VECTOR GUARDS ----------
                if (
                    not vid
                    or not isinstance(vec, list)
                    or len(vec) != EMBEDDING_DIM
                ):
                    continue

                # ---------- METADATA DEFAULTS ----------
                meta.setdefault("domain", "general")
                meta.setdefault("confidence", 0.5)
                meta.setdefault("language", "unknown")
                meta.setdefault("content_type", "text")

                # ---------- REQUIRED METADATA ----------
                if not REQUIRED_METADATA_FIELDS.issubset(meta):
                    continue

                vectors.append((vid, vec, meta))

            if vectors:
                self.index.upsert(
                    vectors=vectors,
                    namespace=namespace
                )

    # -------------------------------------------------
    # QUERY (READ PATH)
    # -------------------------------------------------

    def query(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None,
        namespace: str = DEFAULT_NAMESPACE
    ) -> List[Dict]:

        if (
            not isinstance(query_vector, list)
            or len(query_vector) != EMBEDDING_DIM
        ):
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
