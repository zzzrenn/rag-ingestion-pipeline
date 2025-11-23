import os
import asyncio
import argparse
from dotenv import load_dotenv
from src.rag_client import RAGClient
from openai import AsyncOpenAI

load_dotenv()

async def generate_response(query: str, hybrid_search: bool = False):
    print("Initializing RAG Client...")
    rag_client = RAGClient()
    
    print(f"Retrieving context for: '{query}'")
    async with rag_client.db:  # Properly close DB client
        documents = await rag_client.retrieve(query, limit=5, hybrid_search=hybrid_search)
    
    context_text = "\n\n".join([doc.content for doc in documents])
    print(f"Retrieved {len(documents)} documents.")
    
    print("Generating response with GPT-4o-mini...")
    async with AsyncOpenAI() as client:  # Properly close OpenAI client
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful medical assistant. Answer questions based on the provided context."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_text}\n\nQuestion: {query}\n\nProvide a concise answer based on the context."
                }
            ],
            temperature=0.3,
            max_tokens=500
        )
    
    return response.choices[0].message.content

async def main():
    parser = argparse.ArgumentParser(description="Run RAG generation with specific configuration.")
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

    query = "first line of defense for hypertension in pregnancy."
    answer = await generate_response(query, hybrid_search=args.use_hybrid_search)
    print("\n--- Generated Answer ---")
    print(answer)

if __name__ == "__main__":
    asyncio.run(main())
