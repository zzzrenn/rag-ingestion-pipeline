import os
from typing import List, Dict, Optional
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from src.models import Document, DocMetadata
from src.db.base import BaseVectorDB
from src.db.factory import VectorDBFactory

from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile
)
from azure.search.documents.models import VectorizedQuery

@VectorDBFactory.register("azure")
class AzureAdapter(BaseVectorDB):
    def __init__(self, index_name: str = "rag-index"):
        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        key = os.getenv("AZURE_SEARCH_API_KEY")
        
        if not endpoint or not key:
             pass

        self.index_name = index_name
        self.credential = AzureKeyCredential(key)
        self.endpoint = endpoint
        self._client = None
        self._index_client = None

    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup clients"""
        if self._client:
            await self._client.close()
        if self._index_client:
            await self._index_client.close()

    async def _get_client(self) -> SearchClient:
        """Lazy initialize async search client"""
        if self._client is None:
            await self._ensure_index()
            self._client = SearchClient(
                endpoint=self.endpoint,
                index_name=self.index_name,
                credential=self.credential
            )
        return self._client

    async def _ensure_index(self):
        if self._index_client is None:
            self._index_client = SearchIndexClient(
                endpoint=self.endpoint, 
                credential=self.credential
            )
        
        # Check if index exists
        index_names = [i.name async for i in self._index_client.list_indexes()]
        
        if self.index_name not in index_names:
            print(f"Creating index {self.index_name}...")
            
            # Define Index
            vector_size = int(os.getenv("VECTOR_SIZE", "1536"))
            fields = [
                SimpleField(name="id", type="Edm.String", key=True),
                SearchableField(name="content", type="Edm.String"),
                SearchField(name="embedding", type="Collection(Edm.Single)", vector_search_dimensions=vector_size, vector_search_profile_name="my-vector-config"),
                SimpleField(name="source_type", type="Edm.String", filterable=True),
                SimpleField(name="product", type="Edm.String", filterable=True),
                SimpleField(name="content_type", type="Edm.String", filterable=True),
                SimpleField(name="created_at", type="Edm.DateTimeOffset", filterable=True),
                SimpleField(name="page_number", type="Edm.Int32", filterable=True),
                SimpleField(name="source_filename", type="Edm.String", filterable=True),
                SimpleField(name="parent_id", type="Edm.String", filterable=True),
                SearchableField(name="parent_text", type="Edm.String"),
            ]
            
            vector_search = VectorSearch(
                profiles=[VectorSearchProfile(name="my-vector-config", algorithm_configuration_name="my-hnsw")],
                algorithms=[HnswAlgorithmConfiguration(name="my-hnsw")]
            )
            
            index = SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search)
            await self._index_client.create_index(index)

    async def upsert(self, documents: List[Document], **kwargs):
        client = await self._get_client()
        batch = []
        
        for doc in documents:
            if not doc.embedding:
                continue
                
            # Azure Search expects a flat dictionary
            # Note: sparse_embedding is ignored for Azure
            item = {
                "id": str(doc.id),
                "content": doc.content,
                "embedding": doc.embedding,
                **doc.metadata.model_dump()
            }
            batch.append(item)
            
        if batch:
            await client.upload_documents(documents=batch)

    async def search(self, query_vector: List[float], limit: int = 5, filters: Optional[Dict] = None,
               sparse_query_vector: Optional[dict] = None, search_text: Optional[str] = None) -> List[Document]:
        """
        Search using Azure AI Search
        
        Args:
            query_vector: Dense embedding vector
            limit: Number of results
            filters: Metadata filters
            sparse_query_vector: Ignored for Azure (Qdrant-specific)
            search_text: Optional query text for hybrid search (BM25 + vector)
            
        Returns:
            List of matching documents
        """
        # Note: sparse_query_vector is ignored - Azure uses search_text for hybrid search
        
        client = await self._get_client()
        
        # Construct OData filter
        odata_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                # Simple equality check for now
                conditions.append(f"{key} eq '{value}'")
            odata_filter = " and ".join(conditions)

        results = await client.search(
            search_text=search_text,
            vector_queries=[VectorizedQuery(vector=query_vector, k_nearest_neighbors=limit, fields="embedding")],
            filter=odata_filter,
            top=limit
        )
        
        documents = []
        async for result in results:
            content = result.pop('content')
            embedding = result.pop('embedding', None)
            id_ = result.pop('id')
            score = result.get('@search.score')
            
            # Remove Azure specific fields
            result = {k: v for k, v in result.items() if not k.startswith('@')}
            
            metadata = DocMetadata(**result)
            
            documents.append(Document(
                id=id_,
                content=content,
                metadata=metadata,
                embedding=embedding,
                score=score
            ))
            
        return documents
