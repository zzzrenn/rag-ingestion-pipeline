from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from src.models import Document

class BaseVectorDB(ABC):
    @abstractmethod
    async def upsert(self, documents: List[Document], **kwargs):
        """
        Insert or update documents in the vector database.
        """
        pass

    @abstractmethod
    async def search(
        self, 
        query_vector: List[float], 
        limit: int = 5, 
        filters: Optional[Dict] = None,
        sparse_query_vector: Optional[dict] = None,  # For Qdrant hybrid search
        search_text: Optional[str] = None  # For Azure hybrid search
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query_vector: Dense embedding vector
            limit: Number of results
            filters: Metadata filters
            sparse_query_vector: Sparse BM25 vector for Qdrant hybrid search (ignored by Azure)
            search_text: Query text for Azure hybrid search (ignored by Qdrant)
        """
        pass
