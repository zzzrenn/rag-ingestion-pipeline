import os
import asyncio
import argparse
import json
from dotenv import load_dotenv

load_dotenv()

async def test_ingestion():
    from ingestion_pipeline import aingest_file
    
    parser = argparse.ArgumentParser(description="Test ingestion with a sample PDF")
    parser.add_argument("--db_type", default="qdrant", help="Vector Database Type")
    parser.add_argument("--embedder_type", default="openai", help="Embedder Type")
    args = parser.parse_args()
    
    if args.db_type:
        os.environ["VECTOR_DB_TYPE"] = args.db_type
    if args.embedder_type:
        os.environ["EMBEDDER_TYPE"] = args.embedder_type
    
    # Sample ingestion
    file_path = "sample_data/hypertension-in-pregnancy-diagnosis-and-management-pdf-66141717671365.pdf"
    metadata = {
        "source_type": "pdf",
        "product": "general",
        "content_type": "other"
    }
    
    print(f"Ingesting {file_path}...")
    await aingest_file(file_path, metadata)

if __name__ == "__main__":
    asyncio.run(test_ingestion())
