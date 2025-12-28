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

# HARD SAFETY GUARDS
MIN_RAW_SCORE = 0.22          # ↑ stricter, realistic for cosine
UPSERT_BATCH_SIZE = 100

DEFAULT_NAMESPACE = "agriculture"

REQUIRED_METADATA_FIELDS = {
    "chunk_id",
    "content_type",
    "source",
}

# ---------------- VECTOR STORE ----------------

class VectorStore:
    """
    Pinecone-backed vector store (defensive).
    """

    def __init__(self):
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY not set")

        self.pc = Pinecone(api_key=PINECONE_API_KEY)

        if INDEX_NAME not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric=METRIC,
                spec=ServerlessSpec(
                    cloud="aws",
                    region=PINECONE_ENV
                )
            )
            while not self.pc.describe_index(INDEX_NAME).status["ready"]:
                time.sleep(1)

        self.index = self.pc.Index(INDEX_NAME)

    # -------------------------------------------------
    # UPSERT
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
                if (
                    r.get("id")
                    and isinstance(r.get("vector"), list)
                    and len(r["vector"]) == EMBEDDING_DIM
                ):
                    vectors.append(
                        (r["id"], r["vector"], r.get("metadata", {}))
                    )

            if vectors:
                self.index.upsert(vectors=vectors, namespace=namespace)

    # -------------------------------------------------
    # QUERY (STRICT)
    # -------------------------------------------------

    def query(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None,
        namespace: str = DEFAULT_NAMESPACE
    ) -> List[Dict]:

        if (
            not query_vector
            or not isinstance(query_vector, list)
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
        cleaned = []

        for m in matches:
            score = m.get("score", 0.0)
            meta = m.get("metadata") or {}

            if score < MIN_RAW_SCORE:
                continue

            if not REQUIRED_METADATA_FIELDS.issubset(meta.keys()):
                continue  # 🔑 incomplete chunk → reject

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
