from src.rag_client import RAGClient
from dotenv import load_dotenv

load_dotenv()

def test_retrieval():
    print("Initializing RAG Client...")
    client = RAGClient()
    
    query = "first line of defense for hypertension in pregnancy."
    print(f"Querying: '{query}'")
    
    results = client.retrieve(query, limit=3)
    
    print(f"Found {len(results)} results:")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} (Score: {doc.score}) ---")
        print(f"Content Length: {len(doc.content)} chars")
        print(f"Content: {doc.content[:200]}...")
        print(f"Metadata: {doc.metadata}")

if __name__ == "__main__":
    test_retrieval()
