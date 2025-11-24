from typing import List
import asyncio
from sentence_transformers import SentenceTransformer
from src.models import Document
from src.embedder.base import BaseEmbedder
from src.embedder.factory import EmbedderFactory
from src.utils.logger import logger, time_execution

@EmbedderFactory.register("e5")
class E5Embedder(BaseEmbedder):
    """
    E5 Embedder using sentence-transformers.
    Supports multilingual-e5-small/base/large.
    Handles 'query: ' and 'passage: ' prefixes automatically.
    """
    
    def __init__(self, model_name: str = "./hf-models/e5"):
        self.model_name = model_name
        
        # Determine device: check env var, otherwise let SentenceTransformer auto-detect
        # Common values: "cpu", "cuda", "mps" (for Mac), "cuda:0", etc.
        import os
        import torch
        
        device = os.getenv("EMBEDDING_DEVICE")
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        logger.info(f"Loading E5 model '{model_name}' on device: {device}...")
        
        # Initialize model (this might download it, so it can take time)
        self.model = SentenceTransformer(model_name, device=device, backend="onnx")
    
    @time_execution
    async def embed(self, documents: List[Document], is_query: bool = False) -> List[Document]:
        """
        Generate embeddings using E5 model.
        Adds 'query: ' prefix for queries and 'passage: ' for documents.
        """
        # Prepare texts with appropriate prefix
        prefix = "query: " if is_query else "passage: "
        texts = [f"{prefix}{doc.content}" for doc in documents]
        
        logger.debug(f"Generating embeddings for {len(texts)} documents (is_query={is_query})")
        
        # Run in thread pool as sentence-transformers is sync/CPU-bound
        embeddings = await asyncio.to_thread(
            lambda: self.model.encode(texts, normalize_embeddings=True)
        )
        
        # Assign embeddings back to documents
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding.tolist()
            
        return documents
