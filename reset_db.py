import os
import asyncio
import argparse
from dotenv import load_dotenv

load_dotenv()

async def reset_azure_index(index_name: str = "rag-index"):
    print(f"\n--- Resetting Azure AI Search (Index: {index_name}) ---")
    
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    key = os.getenv("AZURE_SEARCH_API_KEY")

    if not endpoint or not key:
        print("❌ SKIPPING: Azure credentials (ENDPOINT/KEY) not found in .env")
        return

    try:
        # Import inside function to allow running script if Azure SDK is missing
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents.indexes.aio import SearchIndexClient
        from azure.core.exceptions import ResourceNotFoundError

        async with SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(key)) as client:
            try:
                await client.delete_index(index_name)
                print(f"✅ Index '{index_name}' successfully deleted.")
            except ResourceNotFoundError:
                print(f"⚠️  Index '{index_name}' did not exist (nothing to delete).")
            except Exception as e:
                print(f"❌ Error deleting index: {e}")

    except ImportError:
        print("❌ Error: 'azure-search-documents' library is not installed.")
    except Exception as e:
        print(f"❌ Connection/Auth Error: {e}")


async def reset_qdrant_collection(collection_name: str = "rag_collection"):
    print(f"\n--- Resetting Qdrant (Collection: {collection_name}) ---")
    
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url:
        print("❌ SKIPPING: QDRANT_URL not found in .env")
        return

    try:
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.http.exceptions import UnexpectedResponse

        # Initialize client
        client = AsyncQdrantClient(url=url, api_key=api_key)
        
        try:
            # Check existence first to give cleaner feedback
            exists = await client.collection_exists(collection_name)
            
            if exists:
                await client.delete_collection(collection_name)
                print(f"✅ Collection '{collection_name}' successfully deleted.")
            else:
                print(f"⚠️  Collection '{collection_name}' did not exist (nothing to delete).")
                
        except Exception as e:
            print(f"❌ Error during Qdrant operation: {e}")
        finally:
            # Ensure client is closed
            await client.close()

    except ImportError:
        print("❌ Error: 'qdrant-client' library is not installed.")
    except Exception as e:
        print(f"❌ Connection Error: {e}")


async def main():
    parser = argparse.ArgumentParser(description="Reset vector databases")
    parser.add_argument("--db_type", choices=["azure", "qdrant", "all"], default="all", help="Specific DB to reset")
    args = parser.parse_args()

    # Determine what to reset
    reset_azure = args.db_type in ["azure", "all"]
    reset_qdrant = args.db_type in ["qdrant", "all"]

    if reset_azure:
        await reset_azure_index()
    
    if reset_qdrant:
        await reset_qdrant_collection()

if __name__ == "__main__":
    asyncio.run(main())