from qdrant_client import QdrantClient
import pprint

def inspect():
    print("Inspecting QdrantClient...")
    print(f"QdrantClient dir: {dir(QdrantClient)}")
    
    client = QdrantClient(":memory:")
    print(f"Instance dir: {dir(client)}")
    
    if hasattr(client, 'search'):
        print("Instance has 'search'")
    else:
        print("Instance MISSING 'search'")
        
    if hasattr(client, 'query_points'):
        print("Instance has 'query_points'")

if __name__ == "__main__":
    inspect()
