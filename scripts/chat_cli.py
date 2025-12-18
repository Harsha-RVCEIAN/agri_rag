import sys
import traceback

from rag.pipeline import RAGPipeline


def print_header():
    print("\n" + "=" * 60)
    print("🌾 AGRI-RAG — Command Line Interface")
    print("Type your agriculture question.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60 + "\n")


def print_answer(result: dict):
    print("\n🤖 ANSWER:")
    print(result.get("answer", "").strip())

    print("\n📊 CONFIDENCE:")
    print(result.get("confidence", 0.0))

    if result.get("refused"):
        print("\n⚠️  REFUSAL:")
        print("Reason:", result.get("refusal_reason", "unknown"))

    print("\n📚 SOURCES:")
    citations = result.get("citations", [])
    if not citations:
        print("  None")
    else:
        for i, c in enumerate(citations, start=1):
            src = c.get("source", "unknown")
            page = c.get("page", "N/A")
            ctype = c.get("content_type", "text")
            print(f"  {i}. {src} (page {page}, type={ctype})")

    diagnostics = result.get("diagnostics")
    if diagnostics:
        print("\n🧪 DIAGNOSTICS:")
        for k, v in diagnostics.items():
            print(f"  {k}: {v}")

    print("\n" + "-" * 60 + "\n")


def main():
    rag = RAGPipeline()
    print_header()

    while True:
        try:
            query = input("👨‍🌾 Ask: ").strip()

            if not query:
                print("⚠️  Empty question. Try again.\n")
                continue

            if query.lower() in {"exit", "quit"}:
                print("\n👋 Exiting Agri-RAG. Goodbye.")
                break

            # ---------- RAG CALL ----------
            result = rag.run(query=query)

            # ---------- OUTPUT ----------
            print_answer(result)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Exiting cleanly.")
            break

        except Exception as e:
            print("\n❌ SYSTEM ERROR")
            print(str(e))
            print("\nTraceback (for debugging):")
            traceback.print_exc()
            print("\nSystem recovered. You can continue asking questions.\n")


if __name__ == "__main__":
    main()
