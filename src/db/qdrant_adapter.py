import os
from typing import List, Dict, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest
from src.models import Document, DocMetadata
from src.db.base import BaseVectorDB
from src.db.factory import VectorDBFactory
from src.utils.logger import logger, time_execution

@VectorDBFactory.register("qdrant")
class QdrantAdapter(BaseVectorDB):
    def __init__(self, collection_name: str = "rag_collection", use_hybrid: bool = False):
        self.collection_name = collection_name
        self.client = None
        self.use_hybrid = use_hybrid
        
        # Default to localhost if not set
        self.url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = os.getenv("QDRANT_API_KEY")
        
        logger.info(f"Initialized QdrantAdapter for collection '{collection_name}' (hybrid={use_hybrid})")

    async def __aenter__(self):
        """Async context manager entry"""
        await self._get_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup client"""
        if self.client:
            await self.client.close()
            logger.debug("Qdrant client closed")

    async def _get_client(self) -> AsyncQdrantClient:
        """Lazy initialize async client"""
        if self.client is None:
            self.client = AsyncQdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=60.0 # Increase timeout to avoid ResponseHandlingException
            )
            await self._ensure_collection(self.client, self.use_hybrid)
        return self.client

    async def _ensure_collection(self, client: AsyncQdrantClient, use_hybrid: bool):
        """
        Ensure collection exists with appropriate vector configuration
        
        Args:
            use_hybrid: If True, configure collection for hybrid search (dense + sparse)
        """
        collections = await client.get_collections()
        
        if self.collection_name not in [c.name for c in collections.collections]:
            vector_size = int(os.getenv("VECTOR_SIZE", "1536"))
            logger.info(f"Creating collection '{self.collection_name}' with vector_size={vector_size}, hybrid={use_hybrid}")
            
            if use_hybrid:
                # Create collection with both dense and sparse vectors
                await client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE)
                    },
                    sparse_vectors_config={
                        "sparse": rest.SparseVectorParams()
                    }
                )
            else:
                # Dense-only (backward compatible)
                await client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE)
                )

    @time_execution
    async def upsert(self, documents: List[Document], batch_size: int = 64):
        """
        Upsert documents into Qdrant asynchronously with batching
        """
        client = await self._get_client()
        
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
                await client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=False
                )
    
    @time_execution
    async def search(self, query_vector: List[float], limit: int = 5, filters: Optional[Dict] = None, 
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
        
        client = await self._get_client()
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
            # Hybrid search using prefetch with RRF fusion
            results = await client.query_points(
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
            )
            results = results.points
        elif self.use_hybrid:
            # Dense-only on a hybrid collection (use named vector)
            results = await client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using="dense",
                query_filter=query_filter,
                limit=limit
            )
            results = results.points
        else:
            # Dense-only search on single-vector collection (backward compatible)
            results = await client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit
            )
            results = results.points
        
        documents = []
        for hit in results:
            payload = hit.payload
            content = payload.pop('content')
            
            # Reconstruct metadata
            metadata = DocMetadata(**payload)
            
            documents.append(Document(
                id=hit.id,
                content=content,
                metadata=metadata,
                score=hit.score
            ))
            
        logger.info(f"Found {len(documents)} results.")
        return documents
