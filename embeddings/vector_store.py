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


# ---------------- VECTOR STORE ----------------

class VectorStore:
    """
    Pinecone-backed vector store.
    Pinecone configuration is REQUIRED.
    """

    def __init__(self):
        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY not set")

        self.pc = Pinecone(api_key=PINECONE_API_KEY)

        # ---- list index names safely ----
        index_names = self.pc.list_indexes().names()

        if INDEX_NAME not in index_names:
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

    # ---------------- UPSERT ----------------

    def upsert(self, records: List[Dict]) -> None:
        if not records:
            return

        vectors = [
            (r["id"], r["vector"], r["metadata"])
            for r in records
        ]

        self.index.upsert(vectors=vectors)

    # ---------------- DELETE ----------------

    def delete_by_doc(self, doc_id: str) -> None:
        self.index.delete(filter={"doc_id": {"$eq": doc_id}})

    def delete_by_embedding_version(self, embedding_version: str) -> None:
        self.index.delete(
            filter={"embedding_version": {"$eq": embedding_version}}
        )

    # ---------------- QUERY ----------------

    def query(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:

        response = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filters
        )

        matches = response.get("matches", []) or []

        return [
            {
                "id": m.get("id"),
                "score": m.get("score", 0.0),
                "metadata": m.get("metadata", {})
            }
            for m in matches
        ]

    # ---------------- RESET ----------------

    def reset(self) -> None:
        self.index.delete(delete_all=True)
