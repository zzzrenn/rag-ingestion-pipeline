import os
import glob
from ingestion_pipeline import ingest_file

def run():
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
