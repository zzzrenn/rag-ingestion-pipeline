import os
import asyncio
import argparse
import json
from typing import Dict, Any
from dotenv import load_dotenv
import os
import asyncio
import argparse
import json
from typing import Dict, Any
from dotenv import load_dotenv
from src.models import DocMetadata
from src.loader.factory import LoaderFactory
from src.processor.cleaner import SimpleCleaner
from src.processor.chunker import ParentChildChunker
from src.embedder import EmbedderFactory
from src.db import VectorDBFactory
from src.utils.logger import logger, time_execution

load_dotenv()

@time_execution
async def aingest_file(file_path: str, raw_metadata: Dict[str, Any]):
    """
    Async ingestion pipeline:
    1. Validate Metadata
    2. Load Document (Async)
    3. Clean & Chunk
    4. Embed (Async)
    5. Upsert to Vector DB (Async)
    """
    logger.info(f"Starting ingestion for {file_path}...")
    
    # 1. Validate Metadata
    try:
        DocMetadata(**raw_metadata)
        logger.info("Metadata validated.")
    except Exception as e:
        logger.error(f"Metadata validation failed: {e}")
        return

    # 2. Loader (async)
    try:
        loader = LoaderFactory.get_loader(file_path)
        documents = await loader.load(file_path, **raw_metadata)
        logger.info(f"Loaded {len(documents)} pages/documents.")
    except Exception as e:
        logger.error(f"Loading failed: {e}")
        return

    # 3. Processor (Cleaner & Chunker) - can stay sync
    cleaner = SimpleCleaner()
    chunker = ParentChildChunker()
    
    processed_docs = []
    for doc in documents:
        doc.content = cleaner.clean(doc.content)
        chunks = chunker.chunk(doc)
        processed_docs.extend(chunks)
        
    logger.info(f"Created {len(processed_docs)} chunks.")

    # 4. Embedder (async with batching)
    embedder_type = os.getenv("EMBEDDER_TYPE", "openai")
    use_hybrid = os.getenv("USE_HYBRID_SEARCH", "false").lower() == "true"
    db_type = os.getenv("VECTOR_DB_TYPE", "qdrant")
    
    try:
        embedder = EmbedderFactory.create(embedder_type)        
        embedded_docs = await embedder.embed(processed_docs)
        logger.info(f"Embeddings generated using {embedder_type}.")
        
        # Generate sparse embeddings if hybrid search is enabled
        if use_hybrid and db_type == "qdrant":
            try:
                from src.embedder.bm25_embedder import BM25Embedder
                sparse_embedder = BM25Embedder()
                embedded_docs = await sparse_embedder.embed(embedded_docs)
                logger.info("Sparse BM25 embeddings generated.")
            except ImportError:
                logger.warning("fastembed not installed. Skipping sparse embeddings.")
                use_hybrid = False
        
        # Auto-detect vector size from the first embedding
        if embedded_docs and embedded_docs[0].embedding:
            vector_size = len(embedded_docs[0].embedding)
            os.environ["VECTOR_SIZE"] = str(vector_size)
            logger.info(f"Detected vector size: {vector_size}. Set VECTOR_SIZE env var.")
            
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return

    # 5. Database (async)
    try:
        # Pass use_hybrid to Qdrant adapter
        if db_type == "qdrant":
            db = VectorDBFactory.create(db_type, use_hybrid=use_hybrid)
        else:
            db = VectorDBFactory.create(db_type)
        
        async with db:  # Properly close DB client
            await db.upsert(embedded_docs)
        logger.info(f"Documents upserted to database using {db_type}.")
    except Exception as e:
        logger.error(f"Database upsert failed: {e}")
        return

    logger.info("Ingestion complete.")

async def main():
    parser = argparse.ArgumentParser(description="Ingest a document into the RAG system")
    parser.add_argument("file_path", help="Path to the document")
    parser.add_argument("--metadata", help="JSON string of metadata", default='{}')
    
    args = parser.parse_args()
    
    try:
        metadata = json.loads(args.metadata)
    except json.JSONDecodeError:
        logger.error("Invalid JSON metadata")
        return
        
    await aingest_file(args.file_path, metadata)

if __name__ == "__main__":
    asyncio.run(main())
