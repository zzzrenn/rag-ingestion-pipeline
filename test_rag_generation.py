import os
from openai import OpenAI
from dotenv import load_dotenv
from src.rag_client import RAGClient

load_dotenv()

def generate_response(query: str):
    print("Initializing RAG Client...")
    rag_client = RAGClient()
    
    print(f"Retrieving context for: '{query}'")
    documents = rag_client.retrieve(query, limit=3)
    
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

if __name__ == "__main__":
    query = "first line of defense for hypertension in pregnancy."
    answer = generate_response(query)
    print("\n--- Generated Answer ---")
    print(answer)
