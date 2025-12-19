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
MIN_RAW_SCORE = 0.15        # 🔑 stop garbage matches
UPSERT_BATCH_SIZE = 100

DEFAULT_NAMESPACE = "agriculture"

# ---------------- VECTOR STORE ----------------

class VectorStore:
    """
    Pinecone-backed vector store.

    Responsibilities:
    - Index lifecycle
    - Safe upsert
    - Safe query
    - Namespace isolation

    NOT responsible for:
    - Embeddings
    - Ranking logic
    """

    def __init__(self):
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY not set")

        self.pc = Pinecone(api_key=PINECONE_API_KEY)

        existing = self.pc.list_indexes().names()

        if INDEX_NAME not in existing:
            self.pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric=METRIC,
                spec=ServerlessSpec(
                    cloud="aws",
                    region=PINECONE_ENV
                )
            )

            # Wait until index is ready
            while not self.pc.describe_index(INDEX_NAME).status["ready"]:
                time.sleep(1)

        self.index = self.pc.Index(INDEX_NAME)

    # -------------------------------------------------
    # UPSERT (BATCHED + SAFE)
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
                (r["id"], r["vector"], r["metadata"])
                for r in batch
                if r.get("id") and r.get("vector")
            ]

            if vectors:
                self.index.upsert(
                    vectors=vectors,
                    namespace=namespace
                )

    # -------------------------------------------------
    # DELETE
    # -------------------------------------------------

    def delete_by_doc(
        self,
        doc_id: str,
        namespace: str = DEFAULT_NAMESPACE
    ) -> None:
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

    # -------------------------------------------------
    # QUERY (DEFENSIVE)
    # -------------------------------------------------

    def query(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None,
        namespace: str = DEFAULT_NAMESPACE
    ) -> List[Dict]:

        if not query_vector:
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
            # Pinecone transient failure safety
            return []

        matches = response.get("matches") or []

        cleaned = []
        for m in matches:
            score = m.get("score", 0.0)
            if score < MIN_RAW_SCORE:
                continue  # 🔑 garbage guard

            cleaned.append({
                "id": m.get("id"),
                "score": score,
                "metadata": m.get("metadata", {})
            })

        return cleaned

    # -------------------------------------------------
    # RESET (DANGEROUS – DEV ONLY)
    # -------------------------------------------------

    def reset(self, namespace: str = DEFAULT_NAMESPACE) -> None:
        self.index.delete(delete_all=True, namespace=namespace)
