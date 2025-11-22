import os
from typing import Dict, Type
from src.loader.base import BaseLoader
from src.loader.pdf_loader import PDFLoader

class LoaderFactory:
    _loaders: Dict[str, Type[BaseLoader]] = {
        '.pdf': PDFLoader
    }

    @staticmethod
    def get_loader(file_path: str) -> BaseLoader:
        _, ext = os.path.splitext(file_path)
        loader_class = LoaderFactory._loaders.get(ext.lower())
        
        if not loader_class:
            raise ValueError(f"No loader found for extension: {ext}")
            
        return loader_class()
