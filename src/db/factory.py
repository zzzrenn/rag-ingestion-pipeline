from typing import Dict, Type, Any
from src.db.base import BaseVectorDB

class VectorDBFactory:
    _registry: Dict[str, Type[BaseVectorDB]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(db_cls: Type[BaseVectorDB]):
            cls._registry[name] = db_cls
            return db_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseVectorDB:
        if name not in cls._registry:
            raise ValueError(f"VectorDB '{name}' not found. Available: {list(cls._registry.keys())}")
        return cls._registry[name](**kwargs)
