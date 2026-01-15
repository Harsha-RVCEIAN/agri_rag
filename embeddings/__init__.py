"""
Embeddings package.

Handles text embedding and vector store interaction.
"""

from embeddings.embedder import Embedder
from embeddings.vector_store import VectorStore

__all__ = [
    "Embedder",
    "VectorStore",
]
