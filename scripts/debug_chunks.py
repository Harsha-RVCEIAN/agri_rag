import json
from ingestion.pipeline import ingest_pdf

pdf_path = "data/pdfs/crop_insurance.pdf"

chunks = ingest_pdf(pdf_path)

print(f"\nTotal chunks: {len(chunks)}\n")

for i, c in enumerate(chunks[:5]):  # first 5 only
    print("=" * 80)
    print(f"CHUNK {i+1}")
    print("TEXT:\n", c["text"][:500])
    print("\nMETADATA:")
    print(json.dumps(c["metadata"], indent=2))
