from typing import List
import asyncio
from fastembed import SparseTextEmbedding
from src.models import Document
from src.embedder.base import BaseEmbedder
from src.embedder.factory import EmbedderFactory

@EmbedderFactory.register("bm25")
class BM25Embedder(BaseEmbedder):
    """Sparse BM25 embedder using fastembed"""
    
    def __init__(self, model_name: str = "Qdrant/bm25"):
        """
        Initialize BM25 sparse embedder
        
        Args:
            model_name: Fastembed sparse model name (default: "Qdrant/bm25")
        """
        self.model = SparseTextEmbedding(model_name=model_name)
    
    async def embed(self, documents: List[Document], is_query: bool = False) -> List[Document]:
        """
        Generate sparse BM25 embeddings for documents
        
        Args:
            documents: List of Document objects with content
            
        Returns:
            Same documents with sparse_embedding field populated
        """
        texts = [doc.content for doc in documents]
        
        # fastembed doesn't have async support, run in thread pool
        embeddings = await asyncio.to_thread(
            lambda: list(self.model.embed(texts))
        )
        
        # Convert to dict format expected by Qdrant
        for doc, embedding in zip(documents, embeddings):
            # embedding is a SparseEmbedding object with indices and values
            doc.sparse_embedding = {
                "indices": embedding.indices.tolist(),
                "values": embedding.values.tolist()
            }
        
        return documents
