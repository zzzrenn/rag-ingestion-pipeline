import os
from openai import OpenAI
from dotenv import load_dotenv
from src.rag_client import RAGClient

load_dotenv()

def generate_response(query: str, hybrid_search: bool = False):
    print("Initializing RAG Client...")
    rag_client = RAGClient()
    
    print(f"Retrieving context for: '{query}'")
    documents = rag_client.retrieve(query, limit=5, hybrid_search=hybrid_search)
    
    context_text = "\n\n".join([doc.content for doc in documents])
    print(f"Retrieved {len(documents)} documents.")
    
    # Minimal Prompt Template
    system_prompt = "You are a helpful assistant. Use the provided context to answer the user's question."
    user_prompt = f"""Context:
{context_text}

Question: 
{query}

Answer:"""

    print("Generating response with GPT-4o-mini...")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG generation with specific configuration.")
    parser.add_argument("--db_type", default="qdrant", help="Vector Database Type (e.g., qdrant, azure)")
    parser.add_argument("--embedder_type", default="openai", help="Embedder Type (e.g., openai)")
    parser.add_argument("--use_hybrid_search", action="store_true", help="Use hybrid search (e.g., true, false)")
    args = parser.parse_args()

    if args.db_type:
        os.environ["VECTOR_DB_TYPE"] = args.db_type
    if args.embedder_type:
        os.environ["EMBEDDER_TYPE"] = args.embedder_type
    if args.use_hybrid_search:
        os.environ["USE_HYBRID_SEARCH"] = "true"

    query = "first line of defense for hypertension in pregnancy."
    answer = generate_response(query, hybrid_search=args.use_hybrid_search)
    print("\n--- Generated Answer ---")
    print(answer)
