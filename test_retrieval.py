import os
import asyncio
import argparse
from dotenv import load_dotenv
from src.rag_client import RAGClient

load_dotenv()

async def test_retrieval():
    parser = argparse.ArgumentParser(description="Run retrieval with specific configuration.")
    parser.add_argument("--db_type", default="qdrant", help="Vector Database Type (e.g., qdrant, azure)")
    parser.add_argument("--embedder_type", default="openai", help="Embedder Type (e.g., openai)")
    parser.add_argument("--use_hybrid_search", action="store_true", help="Use hybrid search")
    args = parser.parse_args()

    if args.db_type:
        os.environ["VECTOR_DB_TYPE"] = args.db_type
    if args.embedder_type:
        os.environ["EMBEDDER_TYPE"] = args.embedder_type
    if args.use_hybrid_search:
        os.environ["USE_HYBRID_SEARCH"] = "true"

    print("Initializing RAG Client...")
    client = RAGClient()
    
    query = "first line of defense for hypertension in pregnancy."
    print(f"Querying: '{query}'")
    
    async with client.db:  # Properly close DB client
        results = await client.retrieve(query, limit=5, hybrid_search=args.use_hybrid_search)
    
    print(f"Found {len(results)} results:")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Score: {doc.score}")
        print(f"Content Preview: {doc.content[:200]}...")

if __name__ == "__main__":
    asyncio.run(test_retrieval())
