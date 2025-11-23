from abc import ABC, abstractmethod
from typing import List
from src.models import Document

class BaseEmbedder(ABC):
    @abstractmethod
    async def embed(self, documents: List[Document]) -> List[Document]:
        """
        Generate embeddings for a list of documents asynchronously.
        
        Args:
            documents: List of documents to embed
            
        Returns:
            Same documents with embedding field populated
        """
        pass
