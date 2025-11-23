import os
import argparse
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from qdrant_client import QdrantClient

load_dotenv()

def reset_azure(index_name="rag-index"):
    print(f"--- Resetting Azure AI Search (Index: {index_name}) ---")
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    key = os.getenv("AZURE_SEARCH_API_KEY")
    
    if not endpoint or not key:
        print("Error: AZURE_SEARCH_ENDPOINT or AZURE_SEARCH_API_KEY not found in .env")
        return

    try:
        client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(key))
        if index_name in [i.name for i in client.list_indexes()]:
            print(f"Deleting index '{index_name}'...")
            client.delete_index(index_name)
            print("Index deleted.")
        else:
            print(f"Index '{index_name}' does not exist.")
    except Exception as e:
        print(f"Failed to reset Azure: {e}")

def reset_qdrant(collection_name="rag_collection"):
    print(f"--- Resetting Qdrant (Collection: {collection_name}) ---")
    url = os.getenv("QDRANT_URL")
    key = os.getenv("QDRANT_API_KEY")
    
    if not url:
        print("Error: QDRANT_URL not found in .env")
        return

    try:
        client = QdrantClient(url=url, api_key=key)
        collections = client.get_collections()
        if collection_name in [c.name for c in collections.collections]:
            print(f"Deleting collection '{collection_name}'...")
            client.delete_collection(collection_name)
            print("Collection deleted.")
        else:
            print(f"Collection '{collection_name}' does not exist.")
    except Exception as e:
        print(f"Failed to reset Qdrant: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset Vector Databases (Delete Index/Collection).")
    parser.add_argument("--db_type", choices=['azure', 'qdrant', 'all'], default='all', help="Database to reset")
    args = parser.parse_args()

    if args.db_type == 'azure' or args.db_type == 'all':
        reset_azure()
    
    if args.db_type == 'qdrant' or args.db_type == 'all':
        reset_qdrant()
