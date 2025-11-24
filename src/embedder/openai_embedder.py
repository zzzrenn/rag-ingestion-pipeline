from typing import List
import asyncio
from openai import AsyncOpenAI
from src.models import Document
from src.embedder.base import BaseEmbedder
from src.embedder.factory import EmbedderFactory
from src.utils.logger import logger, time_execution

@EmbedderFactory.register("openai")
class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = AsyncOpenAI()
        self.model = model
        logger.info(f"Initialized OpenAIEmbedder with model='{model}'")
    
    @time_execution
    async def embed(self, documents: List[Document], is_query: bool = False) -> List[Document]:
        """
        Generate embeddings asynchronously with batching
        
        Args:
            documents: List of documents to embed
            is_query: Whether this is a query embedding (ignored for OpenAI)
            
        Returns:
            Same documents with embedding field populated
        """
        batch_size = 100
        total_docs = len(documents)
        logger.debug(f"Generating embeddings for {total_docs} documents using OpenAI")
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc.content for doc in batch]
            
            response = await self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            
            for doc, embedding_data in zip(batch, response.data):
                doc.embedding = embedding_data.embedding
                
        return documents
