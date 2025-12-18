# embeddings/vector_store.py

from typing import List, Dict, Optional
import pinecone
import os


# ---------------- CONFIG ----------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agri-rag")

EMBEDDING_DIM = 1024   # bge-large-en-v1.5
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

        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENV
        )

        if INDEX_NAME not in pinecone.list_indexes():
            pinecone.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric=METRIC
            )

        self.index = pinecone.Index(INDEX_NAME)

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

        Returns Pinecone matches with metadata.
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
