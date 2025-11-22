import os
import argparse
import json
from typing import Dict, Any
from dotenv import load_dotenv
from src.models import DocMetadata
from src.loader.factory import LoaderFactory
from src.processor.cleaner import SimpleCleaner
from src.processor.chunker import RecursiveChunker
from src.embedder import EmbedderFactory
from src.db import VectorDBFactory

load_dotenv()

def ingest_file(file_path: str, raw_metadata: Dict[str, Any]):
    print(f"Starting ingestion for {file_path}...")
    
    # 1. Validation
    try:
        metadata = DocMetadata(**raw_metadata)
        print("Metadata validated.")
    except Exception as e:
        print(f"Metadata validation failed: {e}")
        return

    # 2. Loader
    try:
        loader = LoaderFactory.get_loader(file_path)
        documents = loader.load(file_path, **raw_metadata)
        print(f"Loaded {len(documents)} pages/documents.")
    except Exception as e:
        print(f"Loading failed: {e}")
        return

    # 3. Processor (Cleaner & Chunker)
    cleaner = SimpleCleaner()
    chunker = RecursiveChunker()
    
    processed_docs = []
    for doc in documents:
        doc.content = cleaner.clean(doc.content)
        chunks = chunker.chunk(doc)
        processed_docs.extend(chunks)
        
    print(f"Created {len(processed_docs)} chunks.")

    # 4. Embedder
    embedder_type = os.getenv("EMBEDDER_TYPE", "openai")
    try:
        embedder = EmbedderFactory.create(embedder_type)
        embedded_docs = embedder.embed(processed_docs)
        print(f"Embeddings generated using {embedder_type}.")
    except Exception as e:
        print(f"Embedding failed: {e}")
        return

    # 5. Database
    db_type = os.getenv("VECTOR_DB_TYPE", "qdrant")
    try:
        db = VectorDBFactory.create(db_type)
        db.upsert(embedded_docs)
        print(f"Documents upserted to database using {db_type}.")
    except Exception as e:
        print(f"Database upsert failed: {e}")
        return

    print("Ingestion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a file into the RAG system.")
    parser.add_argument("file_path", help="Path to the file to ingest")
    parser.add_argument("--metadata", help="JSON string of metadata", default="{}")
    
    args = parser.parse_args()
    
    try:
        metadata_dict = json.loads(args.metadata)
    except json.JSONDecodeError:
        print("Invalid JSON metadata")
        exit(1)
        
    ingest_file(args.file_path, metadata_dict)
