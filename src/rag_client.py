import os
import asyncio
from typing import List, Dict, Optional
from dotenv import load_dotenv
from src.models import ProductType, ContentType, SourceType, Document, DocMetadata
from src.embedder import EmbedderFactory
from src.db import VectorDBFactory
from src.utils.logger import logger, time_execution

load_dotenv()

class RAGClient:
    def __init__(self, use_hybrid: bool = None):
        """
        Initialize RAG Client
        
        Args:
            use_hybrid: If True, use hybrid search. If None, reads from env.
        """
        embedder_type = os.getenv("EMBEDDER_TYPE", "openai")
        db_type = os.getenv("VECTOR_DB_TYPE", "qdrant")
        
        # Store for later use
        self.db_type = db_type
        
        # Determine hybrid search usage
        if use_hybrid is None:
            use_hybrid = os.getenv("USE_HYBRID_SEARCH", "false").lower() == "true"
        self.use_hybrid = use_hybrid
        
        logger.info(f"Initializing RAG Client with embedder={embedder_type}, db={db_type}, hybrid={use_hybrid}")
        
        self.embedder = EmbedderFactory.create(embedder_type)
        
        # Initialize sparse embedder for Qdrant hybrid search only
        self.sparse_embedder = None
        if self.use_hybrid and db_type == "qdrant":
            try:
                from src.embedder.bm25_embedder import BM25Embedder
                self.sparse_embedder = BM25Embedder()
                logger.info("Initialized BM25 sparse embedder for hybrid search")
            except ImportError:
                logger.error("fastembed not installed. Required for Qdrant hybrid search.")
                raise Exception("fastembed not installed. Required for Qdrant hybrid search.")
        
        # Initialize database adapter
        if db_type == "qdrant":
            self.db = VectorDBFactory.create(db_type, use_hybrid=self.use_hybrid)
        else:
            self.db = VectorDBFactory.create(db_type)

    @time_execution
    async def retrieve(self, query: str, filters: Optional[Dict[str, str]] = None, limit: int = 5, hybrid_search: bool = None):
        """
        Retrieve documents relevant to the query
        
        Args:
            query: Search query text
            filters: Optional metadata filters
            limit: Number of parent documents to return
            hybrid_search: If True/False, override default hybrid search setting
            
        Returns:
            List of parent documents
        """
        logger.info(f"Retrieving for query: '{query}'")
        
        # 1. Sanitize filters
        sanitized_filters = {}
        if filters:
            for key, value in filters.items():
                if key == 'product':
                    if value in ['product_a', 'product_b', 'product_c', 'general']:
                        sanitized_filters[key] = value
                elif key == 'content_type':
                    if value in ['faq', 'description', 'price', 'terms', 'other']:
                        sanitized_filters[key] = value
                elif key == 'source_type':
                    if value in ['pdf', 'markdown']:
                        sanitized_filters[key] = value
            logger.debug(f"Sanitized filters: {sanitized_filters}")
        
        # 2. Embed query (dense vector)
        dummy_doc = Document(content=query, metadata=DocMetadata(source_type='markdown'))
        embedded_docs = await self.embedder.embed([dummy_doc], is_query=True)
        query_vector = embedded_docs[0].embedding
        
        # Auto-detect vector size and set env var for DB adapters
        if query_vector:
            os.environ["VECTOR_SIZE"] = str(len(query_vector))
        
        # 3. Prepare hybrid search parameters based on DB type
        if hybrid_search is None:
            hybrid_search = self.use_hybrid
        
        sparse_query_vector = None
        search_text = None
        
        if hybrid_search:
            if self.db_type == "qdrant":
                # Generate sparse vector for Qdrant
                if self.sparse_embedder:
                    sparse_docs = await self.sparse_embedder.embed([dummy_doc])
                    sparse_query_vector = sparse_docs[0].sparse_embedding
            elif self.db_type == "azure":
                # Pass query text for Azure hybrid search
                search_text = query
        
        # 4. Search (Child Chunks) - both params passed, each DB uses what it needs
        child_docs = await self.db.search(
            query_vector,
            limit=limit * 2,  # Fetch more children to ensure enough parents
            filters=sanitized_filters,
            sparse_query_vector=sparse_query_vector,
            search_text=search_text
        )
        
        # 5. Deduplicate to Parent Chunks
        seen_parents = set()
        parent_docs = []
        
        for doc in child_docs:
            parent_id = doc.metadata.parent_id
            if parent_id and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                
                # Use parent_text as content
                parent_content = doc.metadata.parent_text or doc.content
                new_metadata = doc.metadata.model_copy()
                
                parent_doc = Document(
                    id=parent_id,
                    content=parent_content,
                    metadata=new_metadata,
                    score=doc.score
                )
                parent_docs.append(parent_doc)
                
                if len(parent_docs) >= limit:
                    break
        
        logger.info(f"Retrieved {len(parent_docs)} parent documents")
        return parent_docs
