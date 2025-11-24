from typing import List
import asyncio
from sentence_transformers import SentenceTransformer
from src.models import Document
from src.embedder.base import BaseEmbedder
from src.embedder.factory import EmbedderFactory

@EmbedderFactory.register("e5")
class E5Embedder(BaseEmbedder):
    """
    E5 Embedder using sentence-transformers.
    Supports multilingual-e5-small/base/large.
    Handles 'query: ' and 'passage: ' prefixes automatically.
    """
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        self.model_name = model_name
        
        # Determine device: let SentenceTransformer auto-detect
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
            
        print(f"Loading E5 model '{model_name}' on device: {device}...")
        
        # Initialize model (this might download it, so it can take time)
        self.model = SentenceTransformer(model_name, device=device)
    
    async def embed(self, documents: List[Document], is_query: bool = False) -> List[Document]:
        """
        Generate embeddings using E5 model.
        Adds 'query: ' prefix for queries and 'passage: ' for documents.
        """
        # Prepare texts with appropriate prefix
        prefix = "query: " if is_query else "passage: "
        texts = [f"{prefix}{doc.content}" for doc in documents]
        
        # Run in thread pool as sentence-transformers is sync/CPU-bound
        embeddings = await asyncio.to_thread(
            lambda: self.model.encode(texts, normalize_embeddings=True)
        )
        
        # Assign embeddings back to documents
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding.tolist()
            
        return documents
