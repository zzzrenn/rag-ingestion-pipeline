import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

def check_qdrant():
    url = os.getenv("QDRANT_URL")
    key = os.getenv("QDRANT_API_KEY")
    print(f"Checking connection to {url}...")
    
    try:
        client = QdrantClient(url=url, api_key=key)
        collections = client.get_collections()
        print("Connection successful!")
        print(f"Collections: {collections}")
        if hasattr(client, 'search'):
            print("Client has 'search' method.")
        else:
            print("Client DOES NOT have 'search' method.")
            print(f"Client dir: {dir(client)}")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    check_qdrant()
