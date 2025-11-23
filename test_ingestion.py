import os
import glob
from ingestion_pipeline import ingest_file

import argparse

def run():
    parser = argparse.ArgumentParser(description="Run ingestion with specific configuration.")
    parser.add_argument("--db_type", default="qdrant", help="Vector Database Type (e.g., qdrant, azure)")
    parser.add_argument("--embedder_type", default="openai", help="Embedder Type (e.g., openai)")
    args = parser.parse_args()

    if args.db_type:
        os.environ["VECTOR_DB_TYPE"] = args.db_type
    if args.embedder_type:
        os.environ["EMBEDDER_TYPE"] = args.embedder_type

    pdf_files = glob.glob("sample_data/*.pdf")
    if not pdf_files:
        print("No PDFs found in sample_data")
        return

    metadata = {
        "source_type": "pdf",
        "product": "general",
        "content_type": "other"
    }

    for pdf_path in pdf_files:
        print(f"Ingesting {pdf_path}...")
        ingest_file(pdf_path, metadata)

if __name__ == "__main__":
    run()
