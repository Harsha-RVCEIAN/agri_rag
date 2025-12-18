# embeddings/vector_store.py

from dotenv import load_dotenv
load_dotenv()

from typing import List, Dict, Optional
import os
import time

from pinecone import Pinecone, ServerlessSpec


# ---------------- CONFIG ----------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agri-rag")

# MiniLM embedding dimension
EMBEDDING_DIM = 384
METRIC = "cosine"


# ---------------- VECTOR STORE ----------------

class VectorStore:
    """
    Responsible ONLY for:
    - index creation
    - upsert
    - delete
    - query with metadata filters

    NO embedding logic here.
    """

    def __init__(self):
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not set")

        self.pc = Pinecone(api_key=PINECONE_API_KEY)

        # ---- create index if missing ----
        existing_indexes = [idx["name"] for idx in self.pc.list_indexes()]

        if INDEX_NAME not in existing_indexes:
            self.pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric=METRIC,
                spec=ServerlessSpec(
                    cloud="aws",
                    region=PINECONE_ENV
                )
            )

            # wait until index is ready
            while not self.pc.describe_index(INDEX_NAME).status["ready"]:
                time.sleep(1)

        self.index = self.pc.Index(INDEX_NAME)

    # ---------------- UPSERT ----------------

    def upsert(self, records: List[Dict]) -> None:
        """
        Upsert embedding records.

        Record format:
        {
            "id": str,
            "vector": List[float],
            "metadata": Dict
        }
        """

        if not records:
            return

        vectors = [
            (r["id"], r["vector"], r["metadata"])
            for r in records
        ]

        self.index.upsert(vectors=vectors)

    # ---------------- DELETE ----------------

    def delete_by_doc(self, doc_id: str) -> None:
        """
        Delete all embeddings for a document.
        """
        self.index.delete(
            filter={"doc_id": {"$eq": doc_id}}
        )

    def delete_by_embedding_version(self, embedding_version: str) -> None:
        """
        Delete embeddings from a specific embedding version.
        """
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
        """
        Query vector store with optional metadata filters.
        """

        response = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filters
        )

        return response.get("matches", [])

    # ---------------- RESET ----------------

    def reset(self) -> None:
        """
        Delete ALL vectors from the index.
        USE WITH CAUTION.
        """
        self.index.delete(delete_all=True)
