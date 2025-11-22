from abc import ABC, abstractmethod
from typing import List
from src.models import Document

class BaseLoader(ABC):
    @abstractmethod
    def load(self, file_path: str, **kwargs) -> List[Document]:
        """
        Load a file and return a list of Documents.
        """
        pass
