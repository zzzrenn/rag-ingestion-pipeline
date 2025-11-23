import os
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from src.models import Document, DocMetadata
from src.db.base import BaseVectorDB
from src.db.factory import VectorDBFactory

@VectorDBFactory.register("qdrant")
class QdrantAdapter(BaseVectorDB):
    def __init__(self, collection_name: str = "rag_collection", use_hybrid: bool = False):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.collection_name = collection_name
        self.use_hybrid = use_hybrid
        self._ensure_collection(use_hybrid=use_hybrid)

    def _ensure_collection(self, use_hybrid: bool = False):
        """
        Ensure collection exists with appropriate vector configuration
        
        Args:
            use_hybrid: If True, configure collection for hybrid search (dense + sparse)
        """
        collections = self.client.get_collections()
        if self.collection_name not in [c.name for c in collections.collections]:
            if use_hybrid:
                # Create collection with both dense and sparse vectors
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": rest.VectorParams(size=1536, distance=rest.Distance.COSINE)
                    },
                    sparse_vectors_config={
                        "sparse": rest.SparseVectorParams()
                    }
                )
            else:
                # Dense-only (backward compatible)
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
            
            # Handle hybrid search: use named vectors if sparse embedding exists
            if self.use_hybrid and doc.sparse_embedding:
                vector_dict = {
                    "dense": doc.embedding,
                    "sparse": rest.SparseVector(
                        indices=doc.sparse_embedding["indices"],
                        values=doc.sparse_embedding["values"]
                    )
                }
            else:
                # Dense-only for backward compatibility
                vector_dict = doc.embedding
            
            points.append(rest.PointStruct(
                id=str(doc.id),
                vector=vector_dict,
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

    def search(self, query_vector: List[float], limit: int = 5, filters: Optional[Dict] = None, 
               sparse_query_vector: Optional[dict] = None, search_text: Optional[str] = None) -> List[Document]:
        """
        Search for documents using dense or hybrid (dense + sparse) retrieval
        
        Args:
            query_vector: Dense embedding vector
            limit: Number of results to return
            filters: Metadata filters
            sparse_query_vector: Optional sparse BM25 vector for hybrid search
            search_text: Ignored for Qdrant (used by Azure)
            
        Returns:
            List of matching documents
        """
        # Note: search_text is ignored - Qdrant uses sparse_query_vector for hybrid
        
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

        # Perform hybrid or dense-only search
        if self.use_hybrid and sparse_query_vector:
            # Hybrid search with both dense and sparse using prefetch + RRF
            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    rest.Prefetch(
                        query=query_vector,
                        using="dense",
                        limit=limit * 2
                    ),
                    rest.Prefetch(
                        query=rest.SparseVector(
                            indices=sparse_query_vector["indices"],
                            values=sparse_query_vector["values"]
                        ),
                        using="sparse",
                        limit=limit * 2
                    )
                ],
                query=rest.FusionQuery(fusion=rest.Fusion.RRF),
                query_filter=query_filter,
                limit=limit
            ).points
        elif self.use_hybrid:
            # Dense-only on a hybrid collection (use named vector)
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using="dense",
                query_filter=query_filter,
                limit=limit
            ).points
        else:
            # Dense-only search on single-vector collection (backward compatible)
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
