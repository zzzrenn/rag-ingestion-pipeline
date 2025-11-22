from typing import Dict, Type, Any
from src.embedder.base import BaseEmbedder

class EmbedderFactory:
    _registry: Dict[str, Type[BaseEmbedder]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(embedder_cls: Type[BaseEmbedder]):
            cls._registry[name] = embedder_cls
            return embedder_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseEmbedder:
        if name not in cls._registry:
            raise ValueError(f"Embedder '{name}' not found. Available: {list(cls._registry.keys())}")
        return cls._registry[name](**kwargs)
