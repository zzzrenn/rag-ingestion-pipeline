from abc import ABC, abstractmethod
from typing import List
from src.models import Document

class BaseCleaner(ABC):
    @abstractmethod
    def clean(self, text: str) -> str:
        """
        Clean and normalize text.
        """
        pass

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> List[Document]:
        """
        Split a document into smaller chunks.
        """
        pass
