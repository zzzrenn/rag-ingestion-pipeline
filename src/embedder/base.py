from abc import ABC, abstractmethod
from typing import List
from src.models import Document

class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, documents: List[Document]) -> List[Document]:
        """
        Generate embeddings for a list of documents.
        """
        pass
