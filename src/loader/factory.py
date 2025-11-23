import os
from typing import Dict, Type
from src.loader.base import BaseLoader

class LoaderFactory:
    _loaders: Dict[str, Type[BaseLoader]] = {}

    @classmethod
    def register(cls, extension: str):
        """Decorator to register a loader for a file extension"""
        def decorator(loader_class: Type[BaseLoader]):
            cls._loaders[extension] = loader_class
            return loader_class
        return decorator

    @staticmethod
    def get_loader(file_path: str) -> BaseLoader:
        _, ext = os.path.splitext(file_path)
        loader_class = LoaderFactory._loaders.get(ext.lower())
        
        if not loader_class:
            raise ValueError(f"No loader found for extension: {ext}")
            
        return loader_class()
