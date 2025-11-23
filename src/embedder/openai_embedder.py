from typing import List
import asyncio
from openai import AsyncOpenAI
from src.models import Document
from src.embedder.base import BaseEmbedder
from src.embedder.factory import EmbedderFactory

@EmbedderFactory.register("openai")
class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = AsyncOpenAI()
        self.model = model
    
    async def embed(self, documents: List[Document]) -> List[Document]:
        """
        Generate embeddings asynchronously with batching
        
        Args:
            documents: List of documents to embed
            
        Returns:
            Same documents with embedding field populated
        """
        # Process in batches to avoid rate limits
        batch_size = 100
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc.content for doc in batch]
            
            # Parallel API calls for each batch
            response = await self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            
            # Assign embeddings back to documents
            for doc, embedding_data in zip(batch, response.data):
                doc.embedding = embedding_data.embedding
        
        return documents
