import os
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from src.models import Document, DocMetadata
from src.db.base import BaseVectorDB
from src.db.factory import VectorDBFactory

@VectorDBFactory.register("qdrant")
class QdrantAdapter(BaseVectorDB):
    def __init__(self, collection_name: str = "rag_collection"):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        # Simple check and create if not exists
        # In production, this might be handled separately
        collections = self.client.get_collections()
        if self.collection_name not in [c.name for c in collections.collections]:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(size=1536, distance=rest.Distance.COSINE)
            )

    def upsert(self, documents: List[Document], batch_size: int = 50):
        points = []
        for doc in documents:
            if not doc.embedding:
                continue
                
            payload = doc.metadata.model_dump()
            payload['content'] = doc.content
            
            points.append(rest.PointStruct(
                id=str(doc.id),
                vector=doc.embedding,
                payload=payload
            ))
            
        if points:
            # Batch the upsert
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )

    def search(self, query_vector: List[float], limit: int = 5, filters: Optional[Dict] = None) -> List[Document]:
        query_filter = None
        if filters:
            must_conditions = []
            for key, value in filters.items():
                must_conditions.append(
                    rest.FieldCondition(
                        key=key,
                        match=rest.MatchValue(value=value)
                    )
                )
            query_filter = rest.Filter(must=must_conditions)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit
        ).points
        
        documents = []
        for hit in results:
            payload = hit.payload
            content = payload.pop('content')
            
            # Reconstruct metadata
            # Note: This assumes payload matches DocMetadata structure exactly
            # We might need to handle type conversions if they were lost (e.g. Enums to str)
            metadata = DocMetadata(**payload)
            
            documents.append(Document(
                id=hit.id,
                content=content,
                metadata=metadata,
                score=hit.score
            ))
            
        return documents
