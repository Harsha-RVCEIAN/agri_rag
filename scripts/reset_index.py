import os
import sys

# ---- ensure project root is on path ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from embeddings.vector_store import VectorStore


def confirm() -> bool:
    """
    Explicit human confirmation.
    Prevents accidental data loss.
    """
    print("\n⚠️  WARNING: This will DELETE the entire vector index.")
    print("All embedded documents will be removed.")
    choice = input("Type 'YES' to continue: ").strip()
    return choice == "YES"


def reset_vector_index():
    vector_store = VectorStore()

    if not confirm():
        print("\n❌ Reset cancelled.")
        return

    print("\n🧹 Resetting vector index...")
    vector_store.reset()
    print("✅ Vector index cleared successfully.")


if __name__ == "__main__":
    reset_vector_index()
