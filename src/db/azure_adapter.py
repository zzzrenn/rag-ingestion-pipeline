import os
from typing import List, Dict, Optional
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from src.models import Document, DocMetadata
from src.db.base import BaseVectorDB
from src.db.factory import VectorDBFactory

from azure.search.documents.indexes import SearchIndexClient
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
        
        self._ensure_index()

        self.client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=self.credential
        )

    def _ensure_index(self):
        index_client = SearchIndexClient(endpoint=self.endpoint, credential=self.credential)
        if self.index_name not in [i.name for i in index_client.list_indexes()]:
            # Define Index
            fields = [
                SimpleField(name="id", type="Edm.String", key=True),
                SearchableField(name="content", type="Edm.String"),
                SearchField(name="embedding", type="Collection(Edm.Single)", vector_search_dimensions=1536, vector_search_profile_name="my-vector-config"),
                SimpleField(name="source_type", type="Edm.String", filterable=True),
                SimpleField(name="product", type="Edm.String", filterable=True),
                SimpleField(name="content_type", type="Edm.String", filterable=True),
                SimpleField(name="created_at", type="Edm.DateTimeOffset", filterable=True),
                SimpleField(name="page_number", type="Edm.Int32", filterable=True),
                SimpleField(name="source_filename", type="Edm.String", filterable=True),
            ]
            
            vector_search = VectorSearch(
                profiles=[VectorSearchProfile(name="my-vector-config", algorithm_configuration_name="my-hnsw")],
                algorithms=[HnswAlgorithmConfiguration(name="my-hnsw")]
            )
            
            index = SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search)
            index_client.create_index(index)

    def upsert(self, documents: List[Document], **kwargs):
        batch = []
        for doc in documents:
            if not doc.embedding:
                continue
                
            # Azure Search expects a flat dictionary usually, or specific structure
            # We'll flatten it for simplicity or map it to the index schema
            # Assuming index has fields: id, content, embedding, source_type, product, etc.
            item = {
                "id": str(doc.id),
                "content": doc.content,
                "embedding": doc.embedding,
                **doc.metadata.model_dump()
            }
            batch.append(item)
            
        if batch:
            self.client.upload_documents(documents=batch)

    def search(self, query_vector: List[float], limit: int = 5, filters: Optional[Dict] = None) -> List[Document]:
        # Construct OData filter
        odata_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                # Simple equality check for now
                conditions.append(f"{key} eq '{value}'")
            odata_filter = " and ".join(conditions)

        results = self.client.search(
            search_text=None, # Pure vector search if text is None
            vector_queries=[VectorizedQuery(vector=query_vector, k_nearest_neighbors=limit, fields="embedding")],
            filter=odata_filter,
            top=limit
        )
        
        documents = []
        for result in results:
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
