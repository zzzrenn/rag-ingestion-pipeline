from abc import ABC, abstractmethod
from typing import List
from src.models import Document

class BaseLoader(ABC):
    @abstractmethod
    async def load(self, file_path: str, **kwargs) -> List[Document]:
        """
        Load a file and return a list of Documents asynchronously.
        
        Args:
            file_path: Path to the file to load
            **kwargs: Additional metadata fields
            
        Returns:
            List of Document objects
        """
        pass
