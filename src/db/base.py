from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from src.models import Document

class BaseVectorDB(ABC):
    @abstractmethod
    def upsert(self, documents: List[Document], **kwargs):
        """
        Upsert documents into the database.
        """
        pass

    @abstractmethod
    def search(self, query_vector: List[float], limit: int = 5, filters: Optional[Dict] = None) -> List[Document]:
        """
        Search for similar documents.
        """
        pass
